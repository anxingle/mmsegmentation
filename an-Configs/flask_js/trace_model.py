#!/usr/bin/env python3
"""
使用原始 checkpoint 直接进行 TorchScript 转换
"""

import torch
import torch.nn as nn
import sys
import os

# 添加 mmsegmentation 到路径
sys.path.insert(0, '/data/an/mmsegmentation')

def trace_model_direct():
    try:
        # 首先尝试直接加载 checkpoint
        print("Loading checkpoint...")
        checkpoint_path = '/data/an/mmsegmentation/work_dirs/AN_UNet_middle/epoch_5.pth'

        # 使用 torch.load 直接加载
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"Checkpoint keys: {list(checkpoint.keys())[:5]}")

        # 加载模型架构
        print("Loading model architecture...")
        from mmseg.apis import init_model

        # 临时修复 torch.load 的问题
        original_torch_load = torch.load
        def safe_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        torch.load = safe_torch_load

        try:
            model = init_model(
                '/data/an/mmsegmentation/an-Configs/AN_UNet_middle.py',
                checkpoint_path,
                device='cpu'
            )
            print("Model loaded successfully!")
        finally:
            # 恢复原始 torch.load
            torch.load = original_torch_load

        # 确保所有参数都是 float32
        print("Converting model to float32...")
        model = model.float()
        model.eval()

        # 测试前向传播
        print("Testing forward pass...")
        with torch.no_grad():
            test_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
            try:
                output = model(test_input)
                print(f"Forward pass successful! Output shape: {output.shape}")
            except Exception as e:
                print(f"Forward pass failed: {e}")
                return False

        # 创建包装类
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                # 确保输入是 float32
                if x.dtype != torch.float32:
                    x = x.float()
                return self.model(x)

        wrapper = ModelWrapper(model)

        print("Creating TorchScript trace...")
        example_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

        traced_model = torch.jit.trace(wrapper, example_input, strict=False)
        print("TorchScript trace created successfully!")

        # 保存模型
        output_path = '/data/an/mmsegmentation/an-Configs/flask_js/middle_fixed.pt'
        traced_model.save(output_path)
        print(f"Model saved to: {output_path}")

        # 测试转换后的模型
        print("Testing converted model...")
        loaded_model = torch.jit.load(output_path, map_location='cpu')
        with torch.no_grad():
            test_output = loaded_model(example_input)
            print(f"Test output shape: {test_output.shape}")
            print("✅ Model conversion and test successful!")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = trace_model_direct()
    if success:
        print("🎉 Direct model conversion successful!")
    else:
        print("💥 Model conversion failed!")