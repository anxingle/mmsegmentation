import os
import uuid
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model configuration
MODEL_PATH = '/data/an/mmsegmentation/an-Configs/flask_js/middle.pt'
INPUT_SIZE = (1024, 1024)
MEAN = np.array([83.675, 156.28, 253.53], np.float32) / 255.0
STD = np.array([78.395, 87.12, 5.375], np.float32) / 255.0
NUM_CLASSES = 2

# Load the TorchScript model
print("Loading TorchScript model...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = torch.jit.load(MODEL_PATH, map_location=device)
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_pil):
    """
    Preprocess PIL Image for model inference

    Args:
        image_pil: PIL Image

    Returns:
        torch.Tensor: Preprocessed tensor ready for model (1, 3, H, W)
        tuple: Original image size (height, width)
    """
    # Get original size
    original_size = image_pil.size  # (width, height)

    # Ensure input size is divisible by 32
    target_h, target_w = INPUT_SIZE
    target_h = (target_h // 32) * 32
    target_w = (target_w // 32) * 32

    # Resize image
    image_resized = image_pil.resize((target_w, target_h), Image.LANCZOS)

    # Convert to numpy and normalize
    image_np = np.array(image_resized, np.float32) / 255.0

    # Apply normalization
    image_np = (image_np - MEAN) / STD

    # Convert to tensor: (H, W, C) -> (C, H, W)
    image_tensor = torch.from_numpy(np.transpose(image_np, (2, 0, 1)))

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor, original_size

def process_with_torchscript(image_path):
    """
    Process image using TorchScript model

    Args:
        image_path: Path to input image

    Returns:
        tuple: (original_image, segmentation_mask)
    """
    try:
        # Read and preprocess image
        image_pil = Image.open(image_path).convert('RGB')
        image_tensor, original_size = preprocess_image(image_pil)

        # Move to device
        image_tensor = image_tensor.to(next(model.parameters()).device)

        # Inference
        with torch.no_grad():
            # Get model output
            output = model(image_tensor)

            # Handle different model output formats
            if isinstance(output, (list, tuple)):
                logits = output[0]  # Take first element if it's a list/tuple
            else:
                logits = output

            # Convert to probabilities and get predictions
            if logits.dim() == 4:  # (1, num_classes, H, W)
                probs = F.softmax(logits, dim=1)
                pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy()  # (H, W)
            else:
                raise ValueError(f"Unexpected model output shape: {logits.shape}")

        # Read original image for visualization
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Resize prediction mask to original image size
        pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8),
                                      (original_size[0], original_size[1]),
                                      interpolation=cv2.INTER_NEAREST)

        return original_image, pred_mask_resized

    except Exception as e:
        print(f"Error processing with TorchScript: {e}")
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
    if total_pixels > 0:
        tongue_ratio = (tongue_pixels / total_pixels) * 100
    else:
        tongue_ratio = 0

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
            # Process image using TorchScript model
            original_image, segmentation_mask = process_with_torchscript(filepath)

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

@app.route('/model_info')
def model_info():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    return jsonify({
        'model_type': 'TorchScript',
        'model_path': MODEL_PATH,
        'input_size': INPUT_SIZE,
        'num_classes': NUM_CLASSES,
        'device': str(next(model.parameters()).device),
        'normalization': {
            'mean': MEAN.tolist(),
            'std': STD.tolist()
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)