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
from typing import Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# Normalization parameters copied from AN_UNet_1024_acc_minrun_v1_cut90.py.
# Keeping them as tensors allows direct broadcasting when normalizing inputs.
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


def parse_shape(values) -> Tuple[int, int] | None:
    """Parse `--shape` CLI argument of the form HEIGHT WIDTH."""
    if values is None:
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError('Shape expects two integers: HEIGHT WIDTH.')
    height, width = map(int, values)
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError('Shape dimensions must be positive.')
    return height, width


def load_image(image_path: Path,
               resize_shape: Tuple[int, int] | None,
               match_test_pipeline: bool,
               long_edge: int,
               size_divisor: int) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    """Load an RGB image, optionally resize, pad, and normalize to match training stats."""
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size

    resize_h, resize_w = orig_h, orig_w
    pad_h = pad_w = 0

    if match_test_pipeline:
        if long_edge <= 0:
            raise ValueError('`--long-edge` must be positive when matching test pipeline.')
        if size_divisor <= 0:
            raise ValueError('`--size-divisor` must be positive when matching test pipeline.')
        scale = long_edge / max(orig_h, orig_w)
        if scale != 1.0:
            resize_h = max(int(round(orig_h * scale)), 1)
            resize_w = max(int(round(orig_w * scale)), 1)
        else:
            resize_h, resize_w = orig_h, orig_w
        if image.size != (resize_w, resize_h):
            image = image.resize((resize_w, resize_h), Image.BILINEAR)
    elif resize_shape is not None:
        target_h, target_w = resize_shape
        resize_h, resize_w = target_h, target_w
        # Bilinear resize keeps smooth edges that work well with segmentation models.
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.BILINEAR)

    # Convert to CHW float tensor and normalize with training mean/std.
    np_img = np.asarray(image, dtype=np.float32)

    if match_test_pipeline:
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
        'orig_size': (orig_h, orig_w),
        'resized_size': (resize_h, resize_w),
        'pad': (pad_h, pad_w),
    }

    return tensor, meta


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


def run_inference(model_path: Path,
                  image_path: Path,
                  output_path: Path | None,
                  resize_shape: Tuple[int, int] | None,
                  match_test_pipeline: bool,
                  long_edge: int,
                  size_divisor: int,
                  device: str,
                  threshold: float,
                  binary_colormap: bool) -> Path:
    """End-to-end inference pipeline: load model, preprocess image, run, postprocess."""
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    # Preprocess image and keep track of original resolution for later upsampling.
    tensor, meta = load_image(
        image_path=image_path,
        resize_shape=resize_shape,
        match_test_pipeline=match_test_pipeline,
        long_edge=long_edge,
        size_divisor=size_divisor,
    )
    orig_h, orig_w = meta['orig_size']
    resized_h, resized_w = meta['resized_size']
    pad_h, pad_w = meta['pad']
    tensor = tensor.to(device)

    # TorchScript module returns raw logits; gradients are not required here.
    with torch.no_grad():
        logits = model(tensor)

    if pad_h or pad_w:
        valid_h = logits.shape[2] - pad_h
        valid_w = logits.shape[3] - pad_w
        logits = logits[..., :valid_h, :valid_w]

    # If the image was resized for inference, bring logits back to the original size.
    if (resized_h, resized_w) != (orig_h, orig_w):
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

    num_classes_hint = logits.shape[1]
    mask = logits_to_mask(logits, num_classes_hint, threshold)

    if output_path is None:
        # Default to <image>_mask.png beside the source input.
        output_path = image_path.with_name(f'{image_path.stem}_mask.png')

    save_mask(mask, output_path, binary_colormap=binary_colormap, original_size=(orig_h, orig_w))

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate segmentation mask with TorchScript model.')
    parser.add_argument('--model', required=True, help='Path to TorchScript (.pt) file.')
    parser.add_argument('--image', required=True, help='Input image file (.png/.jpg/.jpeg).')
    parser.add_argument('--output', help='Output mask image path (default: <image>_mask.png).')
    parser.add_argument('--shape', nargs=2, type=int, default=(1024, 1024),
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Resize input to HEIGHT WIDTH before inference (default: 1024 1024).')
    parser.add_argument('--no-resize', action='store_true',
                        help='Skip resizing; use raw image size (requires model traced with dynamic shapes).')
    parser.add_argument('--match-test-pipeline', action='store_true',
                        help='Approximate mmseg test_pipeline: keep ratio to long edge, then pad to size divisor.')
    parser.add_argument('--long-edge', type=int, default=1536,
                        help='Target long edge when --match-test-pipeline is set (default: 1536).')
    parser.add_argument('--size-divisor', type=int, default=32,
                        help='Padding multiple when --match-test-pipeline is set (default: 32).')
    parser.add_argument('--device', default='cpu', help='Inference device, e.g., cpu or cuda:0.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary segmentation when model outputs a single channel.')
    parser.add_argument('--binary-colormap', action='store_true',
                        help='If set, convert foreground mask to 255 (white) for easier visualization.')

    args = parser.parse_args()

    if args.match_test_pipeline and args.no_resize:
        parser.error('--match-test-pipeline cannot be used together with --no-resize.')

    # Determine whether to align inputs with the traced shape or trust native resolution.
    resize_shape = None if args.no_resize else parse_shape(args.shape)
    if args.match_test_pipeline:
        resize_shape = None  # when matching pipeline we derive shape from long edge

    output_path = run_inference(
        model_path=Path(args.model),
        image_path=Path(args.image),
        output_path=Path(args.output) if args.output else None,
        resize_shape=resize_shape,
        match_test_pipeline=args.match_test_pipeline,
        long_edge=args.long_edge,
        size_divisor=args.size_divisor,
        device=args.device,
        threshold=args.threshold,
        binary_colormap=args.binary_colormap,
    )

    # CLI feedback with the destination path.
    print(f'Mask saved to: {output_path}')


if __name__ == '__main__':
    main()
