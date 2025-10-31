"""
Attention: This script converts a model checkpoint from mmpretrain format to a pure PyTorch TorchScript model.
It loads the model using mmpretrain, traces it with a sample input, and saves
"""

import torch
import torch.nn as nn
from mmpretrain import init_model

def extract_pure_pytorch_model(config_path, checkpoint_path, output_path, img_size: tuple):
    """提取为不依赖 mmpretrain 的纯 PyTorch 模型"""
    
    # 加载 mmpretrain 模型
    mmp_model = init_model(config_path, checkpoint_path, device='cpu')
    mmp_model.eval()
    img = torch.rand(1, 3, img_size[0], img_size[1])
    traced = torch.jit.trace(mmp_model, img)  # 假设输入图像大小为 img_size x img_size
    traced.save(output_path)

    return output_path

if __name__ == '__main__':
    config_path = '/data-ssd/workspace/mmpretrain/works/coat_smooth_cls/coat_smooth.py'
    checkpoint_path = '/data-ssd/logs/coat_wet_smooth_dry_0910/epoch_110.pth'
    output_path = '/data-ssd/logs/coat_wet_smooth_dry_0910/coat_smooth_xl_384.pt'
    
    extract_pure_pytorch_model(config_path, checkpoint_path, output_path, (384, 384))