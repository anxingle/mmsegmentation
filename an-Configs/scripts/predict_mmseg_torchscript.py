"""
Inference script for TorchScript-exported MMSegmentation models.

Example:
    python an-Configs/scripts/predict_mmseg_torchscript.py \
        --model work_dirs/AN_UNet_1024_acc_minrun_v1_handcut90/best_mDice_epoch_51.pt \
        --image demo/tongue.jpg \
        --output outputs/tongue_mask.png
"""

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Normalization parameters copied from AN_UNet_1024_acc_minrun_v1_cut90.py.
# Keeping them as tensors allows direct broadcasting when normalizing inputs.
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
DEFAULT_TTA_SCALES: Tuple[float, ...] = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)
LONG_EDGE = 1536
SIZE_DIVISOR = 32


def _preprocess_image(
        image: Image.Image,
        target_size: Tuple[int, int],
        orig_size: Tuple[int, int],
        pad_to_divisor: bool,
        size_divisor: int,
        flip: bool) -> tuple[torch.Tensor, dict[str, tuple[int, int] | bool]]:
    """Resize, optionally flip/pad, normalize, and return tensor plus metadata."""
    resize_h, resize_w = target_size
    if image.size != (resize_w, resize_h):
        image = image.resize((resize_w, resize_h), Image.BILINEAR)

    np_img = np.asarray(image, dtype=np.float32)

    if flip:
        np_img = np_img[:, ::-1, :]

    pad_h = pad_w = 0
    if pad_to_divisor:
        if size_divisor <= 0:
            raise ValueError('`size_divisor` must be positive when padding is enabled.')
        pad_h = (size_divisor - (resize_h % size_divisor)) % size_divisor
        pad_w = (size_divisor - (resize_w % size_divisor)) % size_divisor
        if pad_h or pad_w:
            np_img = np.pad(
                np_img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0,
            )

    tensor = torch.from_numpy(np_img).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - MEAN) / STD

    meta = {
        'orig_size': orig_size,
        'resized_size': (resize_h, resize_w),
        'pad': (pad_h, pad_w),
        'flip': flip,
    }
    return tensor, meta


def _forward_variant(model: torch.jit.ScriptModule,
                     tensor: torch.Tensor,
                     meta: dict[str, tuple[int, int] | bool],
                     device: str) -> torch.Tensor:
    """Forward pass with padding removal, optional flip correction, and resize."""
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)

    pad_h, pad_w = meta['pad']
    if pad_h or pad_w:
        valid_h = logits.shape[2] - pad_h
        valid_w = logits.shape[3] - pad_w
        logits = logits[..., :valid_h, :valid_w]

    if meta.get('flip', False):
        logits = torch.flip(logits, dims=[3])

    orig_h, orig_w = meta['orig_size']
    resized_h, resized_w = meta['resized_size']
    if (resized_h, resized_w) != (orig_h, orig_w):
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    return logits


def logits_to_mask(logits: torch.Tensor,
                   num_classes_hint: int,
                   threshold: float) -> torch.Tensor:
    """Convert raw logits into discrete mask, handling binary vs multi-class cases."""
    if num_classes_hint == 1 or logits.shape[1] == 1:
        # Binary setup: apply sigmoid and threshold to obtain 0/1 mask.
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).to(torch.uint8)
    else:
        # Multi-class: take argmax over channel dimension.
        mask = torch.argmax(logits, dim=1, keepdim=True).to(torch.uint8)
    return mask


def save_mask(mask: torch.Tensor,
              output_path: Path,
              binary_colormap: bool,
              original_size: Tuple[int, int]) -> None:
    """Persist mask tensor as an image resized back to the original resolution."""
    mask_cpu = mask.squeeze(0).cpu()
    if mask_cpu.ndim == 3 and mask_cpu.shape[0] == 1:
        mask_cpu = mask_cpu.squeeze(0)
    mask_np = mask_cpu.numpy()

    # White foreground is handy for binary visualization; otherwise keep class indices.
    if binary_colormap:
        mask_img = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
    else:
        mask_img = Image.fromarray(mask_np.astype(np.uint8), mode='L')

    # Restore original HxW to align with the source image.
    mask_img = mask_img.resize((original_size[1], original_size[0]), Image.NEAREST)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_img.save(output_path)


def _run_tta_inference(model: torch.jit.ScriptModule,
                       image_path: Path,
                       output_path: Path | None,
                       device: str,
                       threshold: float,
                       binary_colormap: bool,
                       size_divisor: int,
                       tta_scales: Sequence[float],
                       long_edge: int) -> Path:
    """Multi-scale + horizontal flip TTA plus test-pipeline resize averaging."""
    base_image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = base_image.size
    orig_size = (orig_h, orig_w)

    logits_sum: torch.Tensor | None = None
    combination_count = 0
    flips = (False, True)

    if long_edge <= 0:
        raise ValueError('`long_edge` must be positive.')
    long_scale = long_edge / max(orig_h, orig_w)
    long_h = max(int(round(orig_h * long_scale)), 1)
    long_w = max(int(round(orig_w * long_scale)), 1)
    tensor, meta = _preprocess_image(
        image=base_image,
        target_size=(long_h, long_w),
        orig_size=orig_size,
        pad_to_divisor=True,
        size_divisor=size_divisor,
        flip=False,
    )
    logits = _forward_variant(model, tensor, meta, device)
    logits_sum = logits if logits_sum is None else logits_sum + logits
    combination_count += 1

    for scale in tta_scales:
        if scale <= 0:
            raise ValueError('All TTA scales must be positive.')
        target_h = max(int(round(orig_h * scale)), 1)
        target_w = max(int(round(orig_w * scale)), 1)

        for flip in flips:
            tensor, meta = _preprocess_image(
                image=base_image,
                target_size=(target_h, target_w),
                orig_size=orig_size,
                pad_to_divisor=True,
                size_divisor=size_divisor,
                flip=flip,
            )
            logits = _forward_variant(model, tensor, meta, device)
            logits_sum = logits if logits_sum is None else logits_sum + logits
            combination_count += 1

    if logits_sum is None or combination_count == 0:
        raise RuntimeError('TTA inference failed to produce any logits.')

    logits = logits_sum / combination_count
    num_classes_hint = logits.shape[1]
    mask = logits_to_mask(logits, num_classes_hint, threshold)

    if output_path is None:
        output_path = image_path.with_name(f'{image_path.stem}_mask.png')

    save_mask(mask, output_path, binary_colormap=binary_colormap, original_size=orig_size)
    return output_path


def run_inference(model_path: Path,
                  image_path: Path,
                  output_path: Path | None,
                  device: str,
                  threshold: float,
                  binary_colormap: bool) -> Path:
    """Load TorchScript model and run fixed multi-scale TTA inference."""
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    return _run_tta_inference(
        model=model,
        image_path=image_path,
        output_path=output_path,
        device=device,
        threshold=threshold,
        binary_colormap=binary_colormap,
        size_divisor=SIZE_DIVISOR,
        tta_scales=DEFAULT_TTA_SCALES,
        long_edge=LONG_EDGE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate segmentation mask with TorchScript model.')
    parser.add_argument('--model', required=True, help='Path to TorchScript (.pt) file.')
    parser.add_argument('--image', required=True, help='Input image file (.png/.jpg/.jpeg).')
    parser.add_argument('--output', help='Output mask image path (default: <image>_mask.png).')
    parser.add_argument('--device', default='cpu', help='Inference device, e.g., cpu or cuda:0.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation when model outputs a single channel.')
    parser.add_argument('--binary-colormap', action='store_true',
                        help='If set, convert foreground mask to 255 (white) for easier visualization.')

    args = parser.parse_args()

    output_path = run_inference(
        model_path=Path(args.model),
        image_path=Path(args.image),
        output_path=Path(args.output) if args.output else None,
        device=args.device,
        threshold=args.threshold,
        binary_colormap=args.binary_colormap,
    )

    print(f'Mask saved to: {output_path}')


if __name__ == '__main__':
    main()
