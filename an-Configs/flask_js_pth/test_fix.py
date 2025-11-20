#!/usr/bin/env python3
"""
测试数据类型修复
"""

import sys
import torch
import numpy as np
import cv2
from app import preprocess_image, postprocess_output, model

def test_data_types():
    print("🔍 测试数据类型修复...")

    # 创建一个测试图像
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite('test_image.png', test_image)

    try:
        # 测试预处理
        print("1. 测试预处理函数...")
        input_tensor = preprocess_image('test_image.png')
        print(f"   输入张量形状: {input_tensor.shape}")
        print(f"   输入张量类型: {input_tensor.dtype}")

        # 测试模型推理
        print("2. 测试模型推理...")
        with torch.no_grad():
            output = model(input_tensor)

        print(f"   输出张量形状: {output.shape}")
        print(f"   输出张量类型: {output.dtype}")

        # 测试后处理
        print("3. 测试后处理函数...")
        pred_mask = postprocess_output(output, (512, 512))
        print(f"   分割掩码形状: {pred_mask.shape}")
        print(f"   分割掩码类型: {pred_mask.dtype}")
        print(f"   唯一值: {np.unique(pred_mask)}")

        print("✅ 数据类型测试通过!")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

    finally:
        # 清理测试文件
        import os
        if os.path.exists('test_image.png'):
            os.remove('test_image.png')

if __name__ == "__main__":
    success = test_data_types()
    sys.exit(0 if success else 1)