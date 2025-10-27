#!/usr/bin/env python3
"""
将舌象分割数据集中的 JSON 标注文件转换为 PNG 掩码图片

处理 datasets/tongue_seg_v0/ann_dir/train 和 datasets/tongue_seg_v0/ann_dir/val 中的 JSON 文件
"""

import json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


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

        return True

    except Exception as e:
        print(f"❌ 转换失败 {json_file_path.name}: {str(e)}")
        return False


def process_directory(directory_path):
    """
    处理指定目录中的所有 JSON 文件

    Args:
        directory_path (Path): 包含 JSON 文件的目录路径
    """
    if not directory_path.exists():
        print(f"❌ 目录不存在: {directory_path}")
        return 0, 0

    # 获取所有 JSON 文件
    json_files = list(directory_path.glob("*.json"))

    if not json_files:
        print(f"⚠️  目录中没有找到 JSON 文件: {directory_path}")
        return 0, 0

    print(f"📁 处理目录: {directory_path}")
    print(f"找到 {len(json_files)} 个 JSON 文件")

    processed_count = 0
    failed_count = 0

    for json_path in json_files:
        try:
            # 生成对应的 PNG 文件路径
            png_filename = json_path.stem + '.png'
            png_path = directory_path / png_filename

            # 检查 PNG 文件是否已存在
            if png_path.exists():
                print(f"⏭️  跳过已存在的文件: {png_filename}")
                processed_count += 1
                continue

            # 转换 JSON 到 PNG
            if convert_json_to_mask(json_path, png_path):
                processed_count += 1
                print(f"✅ 已转换: {json_path.name} -> {png_filename}")
            else:
                failed_count += 1

        except Exception as e:
            print(f"❌ 处理文件失败 {json_path}: {str(e)}")
            failed_count += 1

    return processed_count, failed_count


def main():
    print("=== JSON 标注文件转 PNG 掩码脚本 ===")

    # 定义要处理的目录
    train_dir = Path("datasets/tongue_seg_v0/ann_dir/train")
    val_dir = Path("datasets/tongue_seg_v0/ann_dir/val")

    total_processed = 0
    total_failed = 0

    # 处理训练集
    print("\n🚀 处理训练集标注文件...")
    train_processed, train_failed = process_directory(train_dir)
    total_processed += train_processed
    total_failed += train_failed

    print(f"训练集处理完成: 成功 {train_processed} 个，失败 {train_failed} 个")

    # 处理验证集
    print("\n🚀 处理验证集标注文件...")
    val_processed, val_failed = process_directory(val_dir)
    total_processed += val_processed
    total_failed += val_failed

    print(f"验证集处理完成: 成功 {val_processed} 个，失败 {val_failed} 个")

    # 总结
    print(f"\n=== 转换完成 ===")
    print(f"总计: 成功转换 {total_processed} 个文件，失败 {total_failed} 个文件")
    print(f"输出目录:")
    print(f"  训练集掩码: {train_dir}")
    print(f"  验证集掩码: {val_dir}")


if __name__ == "__main__":
    main()