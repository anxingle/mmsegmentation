#!/usr/bin/env python3
"""
测试直接使用 MMSegmentation 的版本
"""

import sys
import os
import numpy as np
import cv2
import torch
import tempfile

# 添加 mmsegmentation 到路径
sys.path.insert(0, '/data/an/mmsegmentation')

from mmseg.apis import init_model, inference_model

def test_direct_inference():
    print("🧪 测试直接使用 MMSegmentation 进行推理...")
    print("=" * 50)

    try:
        # 1. 加载模型
        print("1. 加载 MMSegmentation 模型...")

        # 临时修复 torch.load 的问题
        original_torch_load = torch.load
        def safe_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = safe_torch_load

        try:
            model = init_model(
                '/data/an/mmsegmentation/an-Configs/AN_UNet_middle.py',
                '/data/an/mmsegmentation/work_dirs/AN_UNet_middle/epoch_5.pth',
                device='cpu'
            )
            print("✅ 模型加载成功")
        finally:
            # 恢复原始 torch.load
            torch.load = original_torch_load

        # 2. 创建测试图像
        print("2. 创建测试图像...")
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 0, 0), -1)

        # 保存测试图像到临时文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            tmp_path = tmp.name

        try:
            print(f"   测试图像保存到: {tmp_path}")

            # 3. 进行推理
            print("3. 进行推理...")
            result = inference_model(model, tmp_path)
            print("✅ 推理成功")

            # 4. 检查结果
            print("4. 检查推理结果...")
            pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
            print(f"   分割掩码形状: {pred_mask.shape}")
            print(f"   分割掩码唯一值: {np.unique(pred_mask)}")
            print("✅ 结果检查成功")

            print("=" * 50)
            print("🎉 直接推理测试通过！")
            return True

        finally:
            # 清理临时文件
            os.unlink(tmp_path)

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_direct_inference()
    sys.exit(0 if success else 1)