# 舌象图片数据处理脚本说明
 写一个预测脚本, 加载 `work_dirs/AN_UNet_1024_acc_minrun_v1_handcut90/best_mDice_epoch_51.pt` 文件, 输入任意图片(`.jpg`, `.png`, `.jpeg` 格式), 可得到图片对应的舌象分类 `mask` 图
  片. 脚本保存在 `an-Configs/scripts` 中. 要求:
  1. 参考 `an-Configs/AN_UNet_1024_acc_minrun_v1_cut90.py` 中 `tta_pipeline`, 新的脚本也加入   `tta_pipeline` 的逻辑来提高分割精度。
## 脚本功能

`process_tongue_dataset.py` 是一个用于处理舌象图片数据的脚本，主要功能包括：

1. **图片处理**: 将 `datasets/demo_datasets` 中的图片（.jpg 和 .png）移动到 `datasets/tongue_seg_v0/img_dir/train`，并统一转换为 .png 格式
2. **标注校验**: 校验 JSON 标注文件是否只有一个 'tf' 标签，且为多边形格式
3. **标注转换**: 将 labelme 格式的 JSON 标注文件转换为 mask 掩码图片，保存到 `datasets/tongue_seg_v0/ann_dir/train`

`convert_mmseg_pth2torchScript.py` 用于将 MMSegmentation 训练得到的 `.pth` 权重导出为不依赖 mmseg/mmcv 环境的 TorchScript `.pt` 文件，便于在纯 PyTorch 环境、C++ 或部署框架中加载推理。

`predict_mmseg_torchscript.py` 则加载上述 TorchScript 文件，对输入的舌象图片生成对应的分割 mask，为后续的分类、分析或可视化提供结果。

## 使用方法

### 1. 查看将要处理的文件（推荐先执行）
```bash
cd an-Configs/scripts
python process_tongue_dataset.py --dry-run
```

### 2. 执行数据处理
```bash
cd an-Configs/scripts
python process_tongue_dataset.py
```

### 3. 导出 TorchScript 模型
```bash
python3 an-Configs/scripts/convert_mmseg_pth2torchScript.py \
  --config an-Configs/AN_UNet_1024_acc_minrun_v1_cut90.py \
  --checkpoint work_dirs/AN_UNet_1024_acc_minrun_v1_handcut90/best_mDice_epoch_51.pth \
  --output work_dirs/AN_UNet_1024_acc_minrun_v1_handcut90/best_mDice_epoch_51.pt \
  --shape 1024 1024 \
  --device cpu
```

参数说明：
- `--config`：MMSegmentation 配置文件路径。
- `--checkpoint`：训练得到的 `.pth` 权重。
- `--output`：导出的 TorchScript `.pt` 文件路径，会自动创建目录。
- `--shape`：导出时使用的输入高度和宽度，需与训练/推理分辨率保持一致。
- `--device`：导出时使用的设备，可选 `cpu` 或 `cuda:0`。

导出过程中脚本会自动处理 `ftfy`、`mmcv` 自定义算子缺失以及 PyTorch 2.6 `weights_only` 导入限制等常见依赖问题。若模型包含动态输入尺寸，请在重新导出前按照需要修改 `--shape` 或重新 tracing 方式。

### 4. 使用 TorchScript 模型进行推理
```bash
python3 an-Configs/scripts/predict_mmseg_torchscript.py \
  --model work_dirs/AN_UNet_1024_acc_minrun_v1_handcut90/best_mDice_epoch_51.pt \
  --image path/to/tongue.jpg \
  --output outputs/tongue_mask.png \
  --binary-colormap
```

参数说明：
- `--model`：TorchScript `.pt` 文件路径。
- `--image`：输入图片路径，支持 `.jpg`、`.jpeg`、`.png`。
- `--output`：输出 mask 路径（默认生成 `<image>_mask.png`）。
- `--shape`：推理前缩放到的尺寸（默认 `1024 1024`，与导出时保持一致）。
- `--no-resize`：若模型支持动态输入，可添加此参数跳过缩放。
- `--device`：推理设备，可选 `cpu` 或 `cuda:0`。
- `--threshold`：模型输出为单通道时的前景阈值（默认 0.5）。
- `--binary-colormap`：将前景像素转为 255，便于快速查看结果。

脚本会按训练时的均值/方差对输入进行归一化，并在生成 mask 后恢复到原始图片分辨率保存。

## 目录结构

脚本执行前后的目录结构：

```
datasets/
├── demo_datasets/              # 原始数据目录
│   ├── 1002.jpg
│   ├── 1002.json
│   └── ...
└── tongue_seg_v0/             # 处理后数据目录
    ├── img_dir/
    │   └── train/            # 处理后的图片（统一为.png格式）
    │       ├── 1002.png
    │       └── ...
    └── ann_dir/
        └── train/            # 转换后的mask标注图片
            ├── 1002.png
            └── ...
```

## 输出说明

脚本执行过程中会显示：

- ✅ 成功处理的文件
- ❌ 处理失败的文件及原因
- 最终的统计信息（成功/失败数量）

## 错误处理

脚本包含以下错误处理机制：

1. **JSON 校验失败**: 如果 JSON 文件不符合要求（标签数量、标签名称、形状类型）
2. **图片转换失败**: 如果图片格式转换出现问题
3. **Mask 生成失败**: 如果多边形数据无效或无法生成 mask
4. **文件读写错误**: 如果文件权限或路径有问题

## 依赖库

脚本需要以下 Python 库：

- `json`, `pathlib`, `argparse` - Python 标准库
- `PIL (Pillow)` - 图片处理
- `numpy` - 数组处理

**注意**: 脚本使用 `pathlib` 进行所有路径操作，提供了更现代化和跨平台的文件处理方式。

安装依赖：
```bash
pip install Pillow numpy
```

## 注意事项

1. 执行前请确保 `datasets/demo_datasets` 目录存在且包含数据
2. 脚本会覆盖目标目录中的同名文件
3. 建议先使用 `--dry-run` 查看将要处理的文件
4. 确保有足够的磁盘空间存储处理后的数据
