#!/usr/bin/env python3
"""
最终测试脚本 - 验证修复后的 Flask 应用
"""

import sys
import os
import numpy as np
import cv2
import torch
from app import preprocess_image, postprocess_output, create_overlay_mask, model

def create_test_image():
    """创建一个测试图像"""
    # 创建一个简单的测试图像 (512x512)
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    # 添加一些简单的图案
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)  # 红色矩形
    cv2.circle(image, (300, 300), 50, (0, 255, 0), -1)  # 绿色圆形

    return image

def test_complete_pipeline():
    print("🧪 测试完整的图像处理流程...")
    print("=" * 50)

    try:
        # 1. 创建测试图像
        print("1. 创建测试图像...")
        test_image = create_test_image()
        cv2.imwrite('test_complete.png', test_image)
        print("✅ 测试图像创建成功")

        # 2. 测试预处理
        print("2. 测试图像预处理...")
        input_tensor = preprocess_image('test_complete.png')
        print(f"   输入张量形状: {input_tensor.shape}")
        print(f"   输入张量类型: {input_tensor.dtype}")
        print("✅ 预处理成功")

        # 3. 测试模型推理
        print("3. 测试模型推理...")
        with torch.no_grad():
            output = model(input_tensor)
        print(f"   输出张量形状: {output.shape}")
        print(f"   输出张量类型: {output.dtype}")
        print("✅ 模型推理成功")

        # 4. 测试后处理
        print("4. 测试后处理...")
        pred_mask = postprocess_output(output, (512, 512))
        print(f"   分割掩码形状: {pred_mask.shape}")
        print(f"   唯一值: {np.unique(pred_mask)}")
        print("✅ 后处理成功")

        # 5. 测试叠加效果
        print("5. 测试叠加效果...")
        overlay = create_overlay_mask(test_image, pred_mask)
        print(f"   叠加图像形状: {overlay.shape}")
        print("✅ 叠加效果生成成功")

        # 6. 保存结果
        print("6. 保存测试结果...")
        cv2.imwrite('test_mask.png', pred_mask * 255)
        cv2.imwrite('test_overlay.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print("✅ 结果保存成功")

        print("=" * 50)
        print("🎉 完整流程测试通过！")
        print("📁 生成的文件:")
        print("   - test_complete.png (原图)")
        print("   - test_mask.png (掩码)")
        print("   - test_overlay.png (叠加效果)")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理临时文件
        for filename in ['test_complete.png', 'test_mask.png', 'test_overlay.png']:
            if os.path.exists(filename):
                os.remove(filename)

if __name__ == "__main__":
    success = test_complete_pipeline()
    sys.exit(0 if success else 1)