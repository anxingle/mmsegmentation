# 舌象图片数据处理脚本说明

## 脚本功能

`process_tongue_dataset.py` 是一个用于处理舌象图片数据的脚本，主要功能包括：

1. **图片处理**: 将 `datasets/demo_datasets` 中的图片（.jpg 和 .png）移动到 `datasets/tongue_seg_v0/img_dir/train`，并统一转换为 .png 格式
2. **标注校验**: 校验 JSON 标注文件是否只有一个 'tf' 标签，且为多边形格式
3. **标注转换**: 将 labelme 格式的 JSON 标注文件转换为 mask 掩码图片，保存到 `datasets/tongue_seg_v0/ann_dir/train`

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