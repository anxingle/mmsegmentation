"""
Convert an MMSegmentation model checkpoint into a TorchScript module.

Usage (example):
    python an-Configs/scripts/convert_mmseg_pth2torchScript.py \
        --config an-Configs/AN_UNet_1024_acc_minrun_v1_cut90.py \
        --checkpoint work_dirs/AN_UNet_1024_acc_minrun_v1_handcut90/best_mDice_epoch_51.pth \
        --output work_dirs/AN_UNet_1024_acc_minrun_v1_handcut90/best_mDice_epoch_51.pt \
        --shape 1024 1024
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import torch

try:
    import ftfy  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - best effort shim
    import importlib.machinery
    import types

    ftfy = types.ModuleType('ftfy')

    def _identity(text):
        return text

    ftfy.fix_text = _identity  # type: ignore[attr-defined]
    ftfy.__spec__ = importlib.machinery.ModuleSpec('ftfy', loader=None)  # type: ignore[attr-defined]
    ftfy.__all__ = ['fix_text']  # type: ignore[attr-defined]
    sys.modules['ftfy'] = ftfy

# Soft-fail shim for mmcv custom ops to avoid import errors when compiled
# extensions are unavailable in the environment used for conversion.
try:
    from mmcv.utils import ext_loader  # type: ignore  # noqa: E402
except ModuleNotFoundError:
    ext_loader = None  # type: ignore[assignment]

if ext_loader is not None:
    import importlib.machinery
    import types

    class _MMCVExtModule(types.ModuleType):

        def __getattr__(self, name):

            def _missing(*args, **kwargs):
                raise NotImplementedError(
                    f'mmcv op `{name}` is unavailable in this conversion environment.')

            return _missing

    def _make_missing(func_name: str):

        def _missing(*args, **kwargs):
            raise NotImplementedError(
                f'mmcv op `{func_name}` is unavailable in this conversion environment.')

        return _missing

    def _fake_load_ext(name, funcs):
        module_name = f'mmcv.{name}'
        module = sys.modules.get(module_name)
        if module is None:
            module = _MMCVExtModule(module_name)
            module.__spec__ = importlib.machinery.ModuleSpec(  # type: ignore[attr-defined]
                module_name, loader=None)
            sys.modules[module_name] = module
        for func in funcs:
            if not hasattr(module, func):
                setattr(module, func, _make_missing(func))
        return module

    ext_loader.load_ext = _fake_load_ext  # type: ignore[assignment]

# Ensure the repository root (which contains `mmseg`) is on PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmseg.apis import init_model  # noqa: E402
try:
    from mmengine.logging.history_buffer import HistoryBuffer  # type: ignore  # noqa: E402
    from numpy import dtype, ndarray  # type: ignore  # noqa: E402
    from numpy.dtypes import Float64DType  # type: ignore  # noqa: E402
    from numpy.core.multiarray import _reconstruct  # type: ignore  # noqa: E402
    from torch.serialization import add_safe_globals  # type: ignore  # noqa: E402

    add_safe_globals([HistoryBuffer, _reconstruct, ndarray, dtype, Float64DType])
except Exception:
    # Fallback silently if APIs are unavailable; torch.load will raise a clearer error.
    pass

# Torch 2.6 defaults to weights_only=True; override to keep legacy behaviour for trusted checkpoints.
_original_torch_load = torch.load


def _torch_load_legacy(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_legacy


def parse_shape(value: Tuple[int, int]) -> Tuple[int, int]:
    if len(value) != 2:
        raise argparse.ArgumentTypeError('Shape must be two integers: HEIGHT WIDTH.')
    height, width = map(int, value)
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError('Shape dimensions must be positive.')
    return height, width


class MMSegForwardWrapper(torch.nn.Module):
    """Wrap MMSegmentation model so it can be scripted with plain tensors."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # The base segmentor returns raw segmentation logits when run in tensor mode.
        return self.model(inputs)  # type: ignore[no-any-return]


def convert_to_torchscript(config: str,
                           checkpoint: str,
                           output: str,
                           shape: Tuple[int, int],
                           device: str = 'cpu') -> Path:
    """Load MMSeg model and export TorchScript file."""
    segmentor = init_model(config, checkpoint, device=device)
    segmentor.eval()

    wrapper = MMSegForwardWrapper(segmentor)

    example = torch.randn(1, 3, shape[0], shape[1], device=device)
    traced = torch.jit.trace(wrapper, example, strict=False)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert MMSeg checkpoint to TorchScript.')
    parser.add_argument('--config', required=True, help='Path to MMSeg config file.')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint (.pth).')
    parser.add_argument('--output', required=True, help='Path to save TorchScript (.pt).')
    parser.add_argument(
        '--shape',
        nargs=2,
        type=int,
        default=(1024, 1024),
        metavar=('HEIGHT', 'WIDTH'),
        help='Input tensor spatial size used during tracing (default: 1024 1024).')
    parser.add_argument('--device', default='cpu', help='Device for tracing (default: cpu).')

    args = parser.parse_args()

    output_path = convert_to_torchscript(
        config=args.config,
        checkpoint=args.checkpoint,
        output=args.output,
        shape=parse_shape(args.shape),
        device=args.device,
    )

    print(f'TorchScript model saved to: {output_path}')


if __name__ == '__main__':
    main()
