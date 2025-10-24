#!/bin/bash

# 舌象数据处理便捷脚本
# Usage:
#   ./process_tongue_data.sh --preview  # Preview files to be processed
#   ./process_tongue_data.sh           # Run actual processing

SCRIPT_DIR="an-Configs/scripts"
SCRIPT_NAME="process_tongue_dataset.py"

# 检查脚本是否存在
if [ ! -f "$SCRIPT_DIR/$SCRIPT_NAME" ]; then
    echo "❌ 错误: 找不到处理脚本 $SCRIPT_DIR/$SCRIPT_NAME"
    exit 1
fi

# 检查数据目录是否存在
if [ ! -d "datasets/demo_datasets" ]; then
    echo "❌ 错误: 找不到数据目录 datasets/demo_datasets"
    exit 1
fi

# 根据参数执行不同的操作
if [ "$1" = "--preview" ]; then
    echo "=== 预览模式 - 查看将要处理的文件 ==="
    python "$SCRIPT_DIR/$SCRIPT_NAME" --dry-run
elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "舌象数据处理脚本"
    echo ""
    echo "用法:"
    echo "  $0 --preview    # 预览将要处理的文件"
    echo "  $0              # 执行数据处理"
    echo "  $0 --help       # 显示帮助信息"
    echo ""
    echo "功能:"
    echo "  1. 将 datasets/demo_datasets 中的图片移动并转换为PNG格式"
    echo "  2. 校验JSON标注文件（检查tf标签）"
    echo "  3. 将JSON标注转换为mask掩码图片"
else
    echo "=== 开始处理舌象数据 ==="
    python "$SCRIPT_DIR/$SCRIPT_NAME"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 数据处理完成！"
        echo "📁 处理后的图片位置: datasets/tongue_seg_v0/img_dir/train"
        echo "📁 处理后的标注位置: datasets/tongue_seg_v0/ann_dir/train"
    else
        echo ""
        echo "❌ 数据处理失败，请检查错误信息"
        exit 1
    fi
fi