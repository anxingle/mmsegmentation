#!/usr/bin/env python3
"""
简化的模型转换脚本 - 直接加载并保存模型，确保数据类型一致
"""

import torch
import torch.nn as nn
from mmseg.apis import init_model

def convert_model():
    print("Loading original model...")
    model = init_model('/data/an/mmsegmentation/an-Configs/AN_UNet_middle.py',
                       '/data/an/mmsegmentation/work_dirs/AN_UNet_middle/epoch_5.pth',
                       device='cpu')

    # 确保模型权重都是 float32
    model = model.float()
    model.eval()

    # 创建包装类来处理输入输出
    class SimpleWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # 确保输入是 float32
            x = x.float()
            return self.model(x)

    wrapped_model = SimpleWrapper(model)

    print("Creating example input...")
    example_input = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)

    print("Tracing model...")
    try:
        traced = torch.jit.trace(wrapped_model, example_input, strict=False)
        print("Model traced successfully!")

        output_path = '/data/an/mmsegmentation/an-Configs/flask_js/middle_fixed.pt'
        traced.save(output_path)
        print(f"Model saved to: {output_path}")

        # 测试转换后的模型
        print("Testing converted model...")
        loaded_model = torch.jit.load(output_path)
        with torch.no_grad():
            output = loaded_model(example_input)
            print(f"Test output shape: {output.shape}")
            print("✅ Model conversion successful!")

        return True

    except Exception as e:
        print(f"❌ Error during tracing: {e}")
        return False

if __name__ == "__main__":
    success = convert_model()
    if success:
        print("🎉 Model conversion completed!")
    else:
        print("💥 Model conversion failed!")