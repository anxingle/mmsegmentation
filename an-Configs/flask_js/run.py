#!/usr/bin/env python3
"""
启动脚本 - 运行图像分割服务
"""

import os
import sys
import subprocess
from pathlib import Path

def check_model():
    """检查模型文件是否存在"""
    model_path = Path("middle.pt")
    if not model_path.exists():
        print("❌ 错误: 找不到 middle.pt 模型文件!")
        print("请确保在运行此脚本之前已经转换了模型文件。")
        return False
    print("✅ 模型文件检查通过")
    return True

def check_dependencies():
    """检查依赖包"""
    try:
        import torch
        import cv2
        import flask
        print("✅ 依赖包检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def main():
    print("🚀 启动图像分割服务...")
    print("-" * 50)

    # 检查模型文件
    if not check_model():
        sys.exit(1)

    # 检查依赖包
    if not check_dependencies():
        sys.exit(1)

    # 启动 Flask 应用
    print("🌐 启动 Flask 服务器...")
    print("📱 前端地址: http://localhost:5000")
    print("🔍 API 地址: http://localhost:5000/upload")
    print("💚 健康检查: http://localhost:5000/health")
    print("-" * 50)
    print("按 Ctrl+C 停止服务器")
    print("-" * 50)

    try:
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()