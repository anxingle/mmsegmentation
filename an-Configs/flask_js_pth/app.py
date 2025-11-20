import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
import sys
import tempfile

# 添加 mmsegmentation 到路径
sys.path.insert(0, '/data/an/mmsegmentation')

from mmseg.apis import init_model, inference_model

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the MMSegmentation model directly
print("Loading MMSegmentation model...")
try:
    # 临时修复 torch.load 的问题
    original_torch_load = torch.load
    def safe_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)
    torch.load = safe_torch_load

    model = init_model(
        '/data/an/mmsegmentation/an-Configs/AN_UNet_middle.py',
        '/data/an/mmsegmentation/work_dirs/AN_UNet_middle/epoch_5.pth',
        device='cpu'
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
finally:
    # 恢复原始 torch.load
    torch.load = original_torch_load

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_with_mmseg(image_path):
    """使用 MMSegmentation 直接处理图像"""
    try:
        # 使用 inference_model 进行推理
        result = inference_model(model, image_path)

        # 获取分割结果
        pred_mask = result.pred_sem_seg.data[0].cpu().numpy()  # (H, W)
        print(f"分割掩码形状: {pred_mask.shape}")
        print(f"分割掩码唯一值: {np.unique(pred_mask)}")

        # 读取原始图像
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_size = (original_image.shape[1], original_image.shape[0])

        # 调整分割结果到原图尺寸
        pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8),
                                      original_size,
                                      interpolation=cv2.INTER_NEAREST)

        return original_image, pred_mask_resized

    except Exception as e:
        print(f"Error processing with MMSegmentation: {e}")
        raise e

def calculate_tongue_ratio(segmentation_mask, original_image_shape):
    """
    计算舌体mask面积占整张图片的比例

    Args:
        segmentation_mask: 分割掩码 (H, W)
        original_image_shape: 原始图像形状 (H, W, C)

    Returns:
        tongue_ratio: 舌体占比百分比 (0-100)
    """
    # 创建二值mask，舌体区域为1 (class 1)，背景为0
    tongue_mask = (segmentation_mask == 1).astype(np.uint8)

    # 计算舌体像素数量
    tongue_pixels = np.sum(tongue_mask)

    # 计算总像素数量
    total_pixels = original_image_shape[0] * original_image_shape[1]  # H * W

    # 计算舌体占比百分比
    tongue_ratio = (tongue_pixels / total_pixels) * 100
    print(f"tongue_ratio: {tongue_ratio}")

    return round(tongue_ratio, 2)

def create_overlay_mask(original_image, segmentation_mask):
    """Create green overlay for segmentation mask"""
    # Create green mask
    green_mask = np.zeros_like(original_image)
    green_mask[:, :, 1] = 255  # Green channel

    # Create binary mask for segmentation (class 1)
    binary_mask = (segmentation_mask == 1).astype(np.float32)

    # Expand dimensions to match image
    binary_mask = np.expand_dims(binary_mask, axis=2)
    binary_mask = np.repeat(binary_mask, 3, axis=2)

    # Create overlay with transparency
    alpha = 0.6  # Transparency factor
    overlay = original_image.astype(np.float32) * (1 - alpha * binary_mask) + \
              green_mask.astype(np.float32) * (alpha * binary_mask)

    return overlay.astype(np.uint8)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = str(uuid.uuid4()) + '_' + filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        try:
            # Process image using MMSegmentation
            original_image, segmentation_mask = process_with_mmseg(filepath)

            # Calculate tongue mask area ratio
            tongue_ratio = calculate_tongue_ratio(segmentation_mask, original_image.shape)

            # Create mask visualization (black and white)
            mask_viz = (segmentation_mask * 255).astype(np.uint8)
            mask_viz = np.stack([mask_viz, mask_viz, mask_viz], axis=2)

            # Create overlay image
            overlay_image = create_overlay_mask(original_image, segmentation_mask)

            # Convert images to base64 for frontend
            def image_to_base64(image):
                _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                img_str = base64.b64encode(buffer).decode('utf-8')
                return f"data:image/png;base64,{img_str}"

            original_base64 = image_to_base64(original_image)
            mask_base64 = image_to_base64(mask_viz)
            overlay_base64 = image_to_base64(overlay_image)

            return jsonify({
                'success': True,
                'original_image': original_base64,
                'mask_image': mask_base64,
                'overlay_image': overlay_base64,
                'tongue_ratio': tongue_ratio
            })

        finally:
            # Clean up uploaded file
            os.remove(filepath)

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)