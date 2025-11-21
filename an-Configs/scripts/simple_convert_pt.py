"""
Simple and reliable MMSegmentation model to TorchScript converter.

Usage:
    python an-Configs/simple_convert.py \
        --config an-Configs/AN_UNet_middle.py \
        --checkpoint work_dirs/AN_UNet_middle/best_mDice_epoch_102.pth \
        --output middle_1024.pt \
        --shape 1024 1024
"""

import argparse
import torch
from pathlib import Path
from mmengine.config import Config

# Handle PyTorch 2.6+ weights_only safety
try:
    from mmengine.logging.history_buffer import HistoryBuffer
    from numpy import dtype, ndarray
    from numpy.dtypes import Float64DType
    from numpy.core.multiarray import _reconstruct
    from torch.serialization import add_safe_globals

    add_safe_globals([HistoryBuffer, _reconstruct, ndarray, dtype, Float64DType])
except Exception:
    pass

# Override torch.load to use weights_only=False for trusted checkpoints
_original_torch_load = torch.load

def _torch_load_legacy(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)

torch.load = _torch_load_legacy

from mmseg.apis import init_model


class MMSegForwardWrapper(torch.nn.Module):
    """Wrap MMSegmentation model for TorchScript export."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning segmentation logits."""
        return self.model(inputs, mode='tensor')


def convert_to_torchscript(config_path, checkpoint_path, output_path, shape=(1024, 1024), device='cpu'):
    """Convert MMSeg checkpoint to TorchScript."""
    print(f"Loading config from: {config_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load model
    model = init_model(config_path, checkpoint_path, device=device)
    model.eval()

    # Wrap model
    wrapper = MMSegForwardWrapper(model)

    # Create example input
    example_input = torch.randn(1, 3, shape[0], shape[1], device=device)

    print(f"Tracing model with input shape: {example_input.shape}")

    # Trace model
    with torch.no_grad():
        traced_model = torch.jit.trace(wrapper, example_input, strict=False)

    # Save model
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving TorchScript model to: {output_path}")
    traced_model.save(str(output_path))

    # Verify
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Conversion successful!")
    print(f"✓ Model size: {file_size_mb:.2f} MB")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert MMSeg checkpoint to TorchScript')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file (.pth)')
    parser.add_argument('--output', required=True, help='Output path for TorchScript model (.pt)')
    parser.add_argument('--shape', nargs=2, type=int, default=[1024, 1024],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='Input shape for tracing (default: 1024 1024)')
    parser.add_argument('--device', default='cpu', help='Device to use (default: cpu)')

    args = parser.parse_args()

    convert_to_torchscript(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        shape=tuple(args.shape),
        device=args.device
    )


if __name__ == '__main__':
    main()
