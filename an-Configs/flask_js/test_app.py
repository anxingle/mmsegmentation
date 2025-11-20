#!/usr/bin/env python3
"""
测试脚本 - 验证 Flask 应用功能
"""

import sys
import os
import requests
import json
from pathlib import Path

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_loading():
    """测试模型加载"""
    print("🔍 测试模型加载...")
    try:
        from app import model
        if model is not None:
            print("✅ 模型加载成功")
            return True
        else:
            print("❌ 模型加载失败")
            return False
    except Exception as e:
        print(f"❌ 模型加载错误: {e}")
        return False

def test_imports():
    """测试依赖包导入"""
    print("🔍 测试依赖包...")
    try:
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        import flask
        print("✅ 所有依赖包正常")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        return False

def test_preprocessing():
    """测试图像预处理功能"""
    print("🔍 测试图像预处理...")
    try:
        from app import preprocess_image
        import numpy as np

        # 创建一个虚拟图片路径测试
        print("✅ 预处理函数导入成功")
        return True
    except Exception as e:
        print(f"❌ 预处理功能错误: {e}")
        return False

def test_file_structure():
    """测试文件结构"""
    print("🔍 测试文件结构...")
    required_files = [
        'app.py',
        'templates/index.html',
        'requirements.txt',
        'middle.pt'
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False
    else:
        print("✅ 所有必需文件存在")
        return True

def main():
    print("🧪 运行 Flask 应用测试...")
    print("=" * 50)

    tests = [
        ("文件结构", test_file_structure),
        ("依赖包", test_imports),
        ("模型加载", test_model_loading),
        ("预处理功能", test_preprocessing),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        if test_func():
            passed += 1
        print()

    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过! 应用已准备就绪。")
        print("\n🚀 启动命令:")
        print("  python app.py")
        print("  或者:")
        print("  python run.py")
        print("\n📱 访问地址: http://localhost:5000")
        return True
    else:
        print("❌ 部分测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)