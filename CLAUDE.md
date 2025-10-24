# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a customized MMSegmentation repository for semantic segmentation projects. MMSegmentation is OpenMMLab's PyTorch-based semantic segmentation toolbox with a modular design that allows easy combination of backbones, heads, and datasets.

**Current Project**: Watermelon semantic segmentation with custom ZihaoDataset implementation.

## Architecture

### Registry System
MMSegmentation uses registries for component management:
- `MODELS`: Backbones, decode heads, segmentors, losses
- `DATASETS`: Dataset implementations (including custom ZihaoDataset)
- `TRANSFORMS`: Data preprocessing and augmentation
- `RUNNERS`: Training runners

### Directory Structure
```
mmseg/
├── models/
│   ├── backbones/       # Feature extractors (ResNet, ViT, Swin, etc.)
│   ├── decode_heads/    # Segmentation heads (FCN, PSP, DeepLab, etc.)
│   ├── segmentors/      # Complete models (EncoderDecoder, CascadeEncoderDecoder)
│   ├── losses/          # Loss functions
│   └── ...
├── datasets/            # Dataset implementations
│   └── ZihaoDataset.py  # Custom watermelon dataset
└── ...

configs/
├── _base_/
│   ├── models/          # Base model configs
│   ├── datasets/        # Dataset and pipeline configs
│   │   └── ZihaoDataset_pipeline.py  # Custom watermelon dataset config
│   ├── schedules/       # Training schedules (20k, 40k, 80k, 160k iterations)
│   └── default_runtime.py
└── [algorithm]/         # Algorithm-specific configs (unet/, deeplabv3plus/, pspnet/, etc.)

tools/
├── train.py             # Training entry point
├── test.py              # Testing/evaluation entry point
└── analysis_tools/      # Analysis utilities
```

### Config Inheritance
Configs use hierarchical inheritance via `_base_` directive:
```python
_base_ = [
    '../_base_/models/fcn_r50-d8.py',
    '../_base_/datasets/ZihaoDataset_pipeline.py',  # Custom dataset
    '../_base_/schedules/schedule_40k.py',
    '../_base_/default_runtime.py'
]
```

## Common Commands

### Training
```bash
# Single GPU training
python tools/train.py <config_file>

# Multi-GPU training (4 GPUs example)
python -m torch.distributed.launch --nproc_per_node=4 tools/train.py <config_file>

# With custom work directory
python tools/train.py <config_file> --work-dir ./work_dirs/custom_exp

# Resume from checkpoint
python tools/train.py <config_file> --resume

# Mixed precision training (AMP)
python tools/train.py <config_file> --amp

# Override config options
python tools/train.py <config_file> --cfg-options optimizer.lr=0.001 train_dataloader.batch_size=4
```

### Testing and Evaluation
```bash
# Basic testing
python tools/test.py <config_file> <checkpoint_file>

# Test with visualization
python tools/test.py <config_file> <checkpoint_file> --show

# Save prediction results
python tools/test.py <config_file> <checkpoint_file> --out ./results

# Test time augmentation (TTA)
python tools/test.py <config_file> <checkpoint_file> --tta
```

### Installation
```bash
# Install MMCV (required dependency)
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"

# Install MMSegmentation in development mode
pip install -v -e .

# Or with specific extras
pip install -v -e .[all]      # All dependencies
pip install -v -e .[tests]    # With test dependencies
pip install -v -e .[optional] # With optional dependencies
```

### Analysis Tools
```bash
# Analyze training logs
python tools/analysis_tools/analyze_logs.py <log_file> --keys loss

# Browse dataset with visualizations
python tools/analysis_tools/browse_dataset.py <config_file>

# Generate confusion matrix
python tools/analysis_tools/confusion_matrix.py <config_file> <checkpoint_file>

# Calculate model FLOPs and parameters
python tools/analysis_tools/get_flops.py <config_file>
```

### Code Quality
```bash
# Format code with YAPF
yapf -r -i mmseg/ tools/ tests/

# Check with flake8
flake8 mmseg/ tools/ tests/

# Run tests
pytest tests/
```

## Custom Dataset: ZihaoDataset

This repository includes a custom dataset implementation for watermelon semantic segmentation:

**Location**: `mmseg/datasets/ZihaoDataset.py`

**Classes**:
- background
- red
- green
- white
- seed-black
- seed-white

**Dataset Structure** (in `Watermelon87_Semantic_Seg_Mask/`):
```
img_dir/
├── train/
└── val/
ann_dir/
├── train/
└── val/
```

**Config**: `configs/_base_/datasets/ZihaoDataset_pipeline.py`
- Input size: 512x512
- Train batch size: 2
- Augmentations: RandomResize, RandomCrop, RandomFlip, PhotoMetricDistortion

## Development Patterns

### Adding a New Dataset
1. Create dataset class in `mmseg/datasets/`:
   ```python
   from mmseg.registry import DATASETS
   from .basesegdataset import BaseSegDataset

   @DATASETS.register_module()
   class MyDataset(BaseSegDataset):
       METAINFO = {
           'classes': ['class1', 'class2', ...],
           'palette': [[R,G,B], [R,G,B], ...]
       }
   ```

2. Register in `mmseg/datasets/__init__.py`

3. Create pipeline config in `configs/_base_/datasets/`

### Adding a New Model Component
```python
from mmseg.registry import MODELS

@MODELS.register_module()
class MyBackbone(nn.Module):
    def __init__(self, ...):
        super().__init__()

    def forward(self, x):
        # implementation
        return x
```

### Config Override Patterns
```python
# In config file
_base_ = './base_config.py'

# Override model settings
model = dict(
    decode_head=dict(num_classes=6)  # For ZihaoDataset
)

# Override training settings
train_dataloader = dict(batch_size=4)
optim_wrapper = dict(optimizer=dict(lr=0.01))
```

## Key Implementation Details

### Segmentation Pipeline
1. **Data Loading**: LoadImageFromFile → LoadAnnotations
2. **Augmentation**: RandomResize → RandomCrop → RandomFlip → PhotoMetricDistortion
3. **Packing**: PackSegInputs (converts to model input format)
4. **Model**: Backbone → Neck (optional) → Decode Head → Output
5. **Loss**: CrossEntropyLoss, DiceLoss, FocalLoss, etc.
6. **Evaluation**: IoUMetric (mIoU, mDice, mFscore)

### Training Flow
- Uses MMEngine's Runner for training orchestration
- Default: iteration-based training (not epoch-based)
- Common schedules: 20k, 40k, 80k, 160k iterations
- Optimizer: AdamW or SGD
- Learning rate scheduler: PolyLR or LinearLR

### Model Components
- **EncoderDecoder**: Standard segmentation model (backbone + decode_head)
- **CascadeEncoderDecoder**: Multi-stage refinement
- **MultimodalEncoderDecoder**: For multimodal inputs (e.g., VPD with text prompts)

## Dependencies

### Core Requirements
- Python >= 3.7
- PyTorch >= 1.6
- MMCV >= 2.0.0rc4, < 2.2.0
- MMEngine >= 0.5.0, < 1.0.0
- MMDetection >= 3.0.0rc0 (for some algorithms)

### Current pyproject.toml includes
- PyTorch Lightning 1.4.2
- timm (for vision transformers)
- transformers (for multimodal models)
- diffusers (for VPD and other generative models)
- Various CV utilities: opencv-python, imageio, kornia, Pillow

## Available Algorithms

**Classic Methods**: FCN, PSPNet, DeepLabV3, DeepLabV3+, UNet
**Attention-based**: DANet, CCNet, OCRNet, DNLNet
**Transformer-based**: SegFormer, Segmenter, SETR, Mask2Former, MaskFormer, K-Net
**Efficient Models**: BiSeNetV1/V2, STDCSeg, Fast-SCNN, ICNet
**Recent**: SAN (CVPR'23), VPD (ICCV'23), PIDNet, DDRNet

## Notebook Workflows

This repository includes Jupyter notebooks for the complete workflow:
- `【C】下载西瓜语义分割数据集.ipynb`: Dataset download
- `【D】可视化探索语义分割数据集.ipynb`: Dataset exploration
- `【E】准备config配置文件-数据集和pipeline.ipynb`: Config preparation
- `【F1-F6】语义分割算法-*.ipynb`: Algorithm-specific training notebooks (UNet, DeepLabV3+, PSPNet, KNet, Segformer, Mask2Former)

## Important Notes

### Working with Custom Datasets
- Always register dataset in `__init__.py` and `__all__`
- Set `num_classes` in decode_head to match dataset classes
- Verify `reduce_zero_label=False` for datasets where class 0 is valid (like ZihaoDataset)

### Config Debugging
```bash
# Print full config
python tools/train.py <config_file> --cfg-options help=True

# Validate config
python -c "from mmengine.config import Config; cfg = Config.fromfile('<config_file>'); print(cfg)"
```

### Common Issues
- **CUDA OOM**: Reduce batch_size or crop_size
- **Dataset not found**: Check data_root path relative to mmsegmentation root
- **Config inheritance errors**: Ensure base configs exist and use correct paths
- **Registry errors**: Verify module is imported and registered with decorator

### Version Migration
This repo uses MMSegmentation 1.x (main branch). For 0.x migration, see `docs/en/migration/`.
