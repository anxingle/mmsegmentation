#!/usr/bin/env python3
"""
最终验证脚本 - 确保完整服务正常工作
"""

import sys
import os
import tempfile
import numpy as np
import cv2
import requests
import base64
import time
import threading
from app import app

def create_test_image_base64():
    """创建测试图像并转换为 base64"""
    # 创建测试图像
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(image, (300, 300), 50, (0, 255, 0), -1)

    # 转换为 base64
    _, buffer = cv2.imencode('.png', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

def run_flask_server():
    """在后台运行 Flask 服务器"""
    app.run(debug=False, host='127.0.0.1', port=5001, use_reloader=False)

def test_api():
    """测试 API 接口"""
    print("🧪 测试图像分割 API...")

    # 等待服务器启动
    time.sleep(3)

    try:
        # 测试健康检查
        print("1. 测试健康检查接口...")
        response = requests.get('http://127.0.0.1:5001/health', timeout=5)
        if response.status_code == 200:
            print(f"✅ 健康检查成功: {response.json()}")
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False

        # 测试图像上传
        print("2. 测试图像上传接口...")

        # 创建测试图像文件
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, test_image)
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'rb') as f:
                files = {'file': f}
                response = requests.post('http://127.0.0.1:5001/upload', files=files, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("✅ 图像分割成功!")
                    print(f"   原图数据长度: {len(result.get('original_image', ''))}")
                    print(f"   掩码数据长度: {len(result.get('mask_image', ''))}")
                    print(f"   叠加数据长度: {len(result.get('overlay_image', ''))}")
                    return True
                else:
                    print(f"❌ 分割失败: {result.get('error', '未知错误')}")
                    return False
            else:
                print(f"❌ 上传失败: HTTP {response.status_code}")
                print(f"   错误信息: {response.text}")
                return False

        finally:
            os.unlink(tmp_path)

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试错误: {e}")
        return False

def main():
    print("🚀 启动最终验证测试...")
    print("=" * 50)

    # 启动 Flask 服务器线程
    server_thread = threading.Thread(target=run_flask_server, daemon=True)
    server_thread.start()
    print("📡 Flask 服务器已启动...")

    # 运行 API 测试
    success = test_api()

    print("=" * 50)
    if success:
        print("🎉 所有测试通过！图像分割服务已准备就绪。")
        print("\n📱 访问地址: http://127.0.0.1:5001")
        print("🔍 API 接口: http://127.0.0.1:5001/upload")
        print("💚 健康检查: http://127.0.0.1:5001/health")
    else:
        print("💥 测试失败，请检查配置。")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)