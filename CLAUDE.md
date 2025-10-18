# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MMSegmentation is OpenMMLab's semantic segmentation toolbox built on PyTorch. It provides a unified benchmark for various semantic segmentation methods with a modular design that allows easy combination of different components.

## Core Architecture

### Registry-Based System
MMSegmentation uses a registry pattern for extensibility:
- `MODELS` registry for backbones, decode heads, segmentors, losses, etc.
- `DATASETS` registry for dataset implementations
- `TRANSFORMS` registry for data preprocessing

### Config Structure
- `configs/_base_/models/` - Base model configurations
- `configs/_base_/datasets/` - Dataset configurations
- `configs/_base_/schedules/` - Training schedules
- `configs/_base_/default_runtime.py` - Default runtime settings
- Model configs inherit from base configs using `_base_` directive

### Key Components
- **Backbones** (`mmseg/models/backbones/`): Feature extractors (ResNet, ViT, Swin, etc.)
- **Decode Heads** (`mmseg/models/decode_heads/`): Segmentation heads
- **Segmentors** (`mmseg/models/segmentors/`): Complete models (EncoderDecoder, etc.)
- **Losses** (`mmseg/models/losses/`): Loss functions for training
- **Datasets** (`mmseg/datasets/`): Dataset implementations

## Common Development Commands

### Training
```bash
# Single GPU training
python tools/train.py <config_file>

# Multi-GPU training
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py <config_file>

# With custom work directory
python tools/train.py <config_file> --work-dir ./work_dirs/custom_exp

# Resume from checkpoint
python tools/train.py <config_file> --resume

# Mixed precision training
python tools/train.py <config_file> --amp

# Override config options
python tools/train.py <config_file> --cfg-options optimizer.lr=0.001
```

### Testing
```bash
# Basic testing
python tools/test.py <config_file> <checkpoint_file>

# Test with visualization
python tools/test.py <config_file> <checkpoint_file> --show

# Save prediction results
python tools/test.py <config_file> <checkpoint_file> --out ./results

# Test time augmentation
python tools/test.py <config_file> <checkpoint_file> --tta
```

### Installation
```bash
# Install with all dependencies
pip install -e .[all]

# Install minimal runtime dependencies
pip install -e .[runtime]

# Install with test dependencies
pip install -e .[tests]
```

### Code Quality
```bash
# Install pre-commit hooks
pre-commit install

# Run linting manually
flake8 mmseg/ tools/ tests/
yapf -r -i mmseg/ tools/ tests/
isort mmseg/ tools/ tests/

# Run tests
pytest tests/
```

### Dataset Preparation
```bash
# Convert datasets using provided tools
python tools/dataset_converters/cityscapes.py
python tools/dataset_converters/ade20k.py
python tools/dataset_converters/pascal_context.py
```

## Development Patterns

### Config Inheritance
Configs use hierarchical inheritance. Example:
```python
_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py'
]
```

### Model Registration
New models are registered via decorators:
```python
from mmseg.registry import MODELS

@MODELS.register_module()
class MyBackbone(nn.Module):
    # implementation
```

### Working with Components
- Backbones are defined in `model['backbone']`
- Decode heads in `model['decode_head']`
- Auxiliary heads in `model['auxiliary_head']`
- Loss functions in `model['decode_head']['loss_decode']`

## Important Notes

### Version Compatibility
- Requires PyTorch 1.6+
- MMCV >= 2.0.0rc4, < 2.2.0
- MMEngine >= 0.5.0, < 1.0.0

### Common Loss Functions
- CrossEntropyLoss: Standard classification loss
- DiceLoss: Dice coefficient loss
- LovaszLoss: Lovász-Softmax loss
- FocalLoss: Focal loss for class imbalance

### Available Datasets
MMSegmentation supports 20+ datasets including:
- Cityscapes, ADE20K, Pascal VOC
- COCO-Stuff, Mapillary Vistas
- Medical imaging datasets (CHASE_DB1, DRIVE, etc.)

### Analysis Tools
- `tools/analysis_tools/analyze_logs.py` - Training log analysis
- `tools/analysis_tools/confusion_matrix.py` - Confusion matrix visualization
- `tools/analysis_tools/get_flops.py` - Model FLOPs computation
- `tools/analysis_tools/browse_dataset.py` - Dataset visualization

### Model Conversion
Converters available for:
- Vision Transformer models (`tools/model_converters/vit2mmseg.py`)
- Swin Transformer (`tools/model_converters/swin2mmseg.py`)
- BEiT (`tools/model_converters/beit2mmseg.py`)
- And many others