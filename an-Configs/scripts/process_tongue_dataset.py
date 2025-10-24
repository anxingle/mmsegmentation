#!/usr/bin/env python3
"""
舌象图片数据处理脚本

功能:
1. 将 datasets/demo_datasets 中的图片移动到 datasets/tongue_seg_v0/img_dir/train 并转换 .jpg 为 .png
2. 校验 JSON 标注文件是否只有一个 'tf' 标签
3. 将 JSON 标注文件转换为 mask 掩码图片并保存到 datasets/tongue_seg_v0/ann_dir/train

作者: Claude
日期: 2024
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import argparse


def create_directory_structure():
    """创建目标目录结构"""
    img_dir = Path("datasets/tongue_seg_v0/img_dir/train")
    ann_dir = Path("datasets/tongue_seg_v0/ann_dir/train")

    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("目录结构已创建/确认")


def validate_json_annotation(json_file_path):
    """
    校验 JSON 文件是否只有一个 'tf' 标签

    Args:
        json_file_path (Path): JSON 文件路径

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get('shapes', [])

        # 检查是否只有一个形状
        if len(shapes) != 1:
            return False, f"标签数量错误: 期望1个，实际{len(shapes)}个"

        # 检查标签是否为 'tf'
        label = shapes[0].get('label', '')
        if label != 'tf':
            return False, f"标签名称错误: 期望'tf'，实际'{label}'"

        # 检查形状类型是否为多边形
        shape_type = shapes[0].get('shape_type', '')
        if shape_type != 'polygon':
            return False, f"形状类型错误: 期望'polygon'，实际'{shape_type}'"

        return True, "校验通过"

    except json.JSONDecodeError as e:
        return False, f"JSON格式错误: {str(e)}"
    except Exception as e:
        return False, f"校验异常: {str(e)}"


def convert_json_to_mask(json_file_path, mask_output_path):
    """
    将 JSON 标注文件转换为 mask 掩码图片

    Args:
        json_file_path (Path): JSON 文件路径
        mask_output_path (Path): 输出 mask 图片路径
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取图片尺寸
        image_width = data.get('imageWidth', 0)
        image_height = data.get('imageHeight', 0)

        if image_width == 0 or image_height == 0:
            raise ValueError("图片尺寸无效")

        # 创建空白 mask (单通道灰度图)
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # 获取多边形坐标
        shapes = data.get('shapes', [])
        if len(shapes) > 0:
            shape = shapes[0]
            points = shape.get('points', [])

            if len(points) >= 3:  # 至少需要3个点构成多边形
                # 转换为整数坐标
                polygon_points = [(int(x), int(y)) for x, y in points]

                # 使用 PIL 绘制多边形
                mask_img = Image.fromarray(mask)
                draw = ImageDraw.Draw(mask_img)
                draw.polygon(polygon_points, fill=1)  # 1 表示前景标签物体
                mask = np.array(mask_img)

        # 保存 mask 图片
        mask_img = Image.fromarray(mask)
        mask_img.save(mask_output_path)

    except Exception as e:
        raise RuntimeError(f"转换 JSON 到 mask 失败: {str(e)}")


def process_images():
    """
    处理图片文件: 移动并转换格式
    """
    source_dir = Path("datasets/demo_datasets")
    target_dir = Path("datasets/tongue_seg_v0/img_dir/train")

    # 获取所有图片文件 (.jpg 和 .png)
    image_files = list(source_dir.glob("*.jpg")) + list(source_dir.glob("*.png"))

    processed_count = 0

    for img_path in image_files:
        try:
            name_without_ext = img_path.stem  # 获取文件名（不含扩展名）
            target_file = target_dir / f"{name_without_ext}.png"  # 统一为PNG格式

            # 如果是 .jpg 文件，需要转换格式
            if img_path.suffix.lower() == '.jpg':
                with Image.open(img_path) as img:
                    img.save(target_file, 'PNG')
            else:
                # 已经是 .png 文件，直接复制
                target_file.write_bytes(img_path.read_bytes())

            processed_count += 1
            print(f"已处理图片: {img_path.name} -> {name_without_ext}.png")

        except Exception as e:
            print(f"处理图片失败 {img_path}: {str(e)}")

    return processed_count


def process_annotations():
    """
    处理标注文件: 校验并转换为 mask
    """
    source_dir = Path("datasets/demo_datasets")
    target_dir = Path("datasets/tongue_seg_v0/ann_dir/train")

    # 获取所有 JSON 文件
    json_files = list(source_dir.glob("*.json"))

    processed_count = 0
    failed_count = 0

    for json_path in json_files:
        try:
            name_without_ext = json_path.stem  # 获取文件名（不含扩展名）
            target_mask_path = target_dir / f"{name_without_ext}.png"

            # 校验 JSON 文件
            is_valid, error_msg = validate_json_annotation(json_path)

            if not is_valid:
                print(f"❌ JSON 校验失败 {json_path.name}: {error_msg}")
                failed_count += 1
                continue

            # 转换为 mask
            convert_json_to_mask(json_path, target_mask_path)

            processed_count += 1
            print(f"✅ 已处理标注: {json_path.name} -> {name_without_ext}.png")

        except Exception as e:
            print(f"❌ 处理标注失败 {json_path}: {str(e)}")
            failed_count += 1

    return processed_count, failed_count


def main():
    parser = argparse.ArgumentParser(description="舌象图片数据处理脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要处理的文件，不执行实际操作")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN 模式 ===")
        source_dir = Path("datasets/demo_datasets")
        print(f"将要处理的图片文件:")

        # 使用 pathlib.glob() 获取图片文件
        for ext in ['*.jpg', '*.png']:
            files = list(source_dir.glob(ext))
            for f in files:
                print(f"  {f}")

        print(f"将要处理的标注文件:")
        json_files = list(source_dir.glob("*.json"))
        for f in json_files:
            print(f"  {f}")
        return

    print("=== 舌象图片数据处理开始 ===")

    # 1. 创建目录结构
    create_directory_structure()

    # 2. 处理图片文件
    print("\n1. 处理图片文件...")
    image_count = process_images()
    print(f"图片处理完成，共处理 {image_count} 个文件")

    # 3. 处理标注文件
    print("\n2. 处理标注文件...")
    annotation_count, failed_count = process_annotations()
    print(f"标注处理完成，成功处理 {annotation_count} 个文件，失败 {failed_count} 个文件")

    print("\n=== 数据处理完成 ===")
    print(f"总结:")
    print(f"  图片文件: {image_count} 个")
    print(f"  标注文件: {annotation_count} 个成功, {failed_count} 个失败")
    print(f"  输出目录:")
    print(f"    图片: datasets/tongue_seg_v0/img_dir/train")
    print(f"    标注: datasets/tongue_seg_v0/ann_dir/train")


if __name__ == "__main__":
    main()