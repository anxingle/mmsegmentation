import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import io
import base64
import json

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the TorchScript model
print("Loading TorchScript model...")
try:
    # Try to load the fixed model first
    model = torch.jit.load('middle_fixed.pt', map_location='cpu')
    model.eval()
    print("Fixed model loaded successfully!")
except Exception as e:
    print(f"Could not load fixed model: {e}")
    try:
        # Fall back to the original model
        model = torch.jit.load('middle.pt', map_location='cpu')
        model.eval()
        print("Original model loaded successfully!")
    except Exception as e2:
        print(f"Error loading model: {e2}")
        model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(1024, 1024)):
    """Preprocess image for model inference"""
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to target size
    image = cv2.resize(image, target_size)

    # Normalize according to config values
    mean = np.array([83.675, 156.28, 253.53])
    std = np.array([78.395, 87.12, 5.375])
    image = image.astype(np.float32)
    image = (image - mean) / std

    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    return image_tensor

def postprocess_output(output_tensor, original_size):
    """Convert model output to segmentation mask"""
    # Get predicted class for each pixel
    if output_tensor.dim() == 4:  # Batch dimension present
        output_tensor = output_tensor.squeeze(0)

    # Convert to float32 if it's in double format
    if output_tensor.dtype == torch.float64:
        output_tensor = output_tensor.float()

    # Get argmax across channel dimension
    pred_mask = torch.argmax(output_tensor, dim=0).cpu().numpy()

    # Resize back to original image size
    pred_mask_resized = cv2.resize(pred_mask.astype(np.uint8),
                                  original_size,
                                  interpolation=cv2.INTER_NEAREST)

    return pred_mask_resized

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

        # Get original image for overlay
        original_image = cv2.imread(filepath)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        original_size = (original_image.shape[1], original_image.shape[0])

        # Preprocess and run inference
        input_tensor = preprocess_image(filepath)

        with torch.no_grad():
            output = model(input_tensor)

        # Postprocess to get segmentation mask
        segmentation_mask = postprocess_output(output, original_size)

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

        # Also convert original image
        original_base64 = image_to_base64(original_image)
        mask_base64 = image_to_base64(mask_viz)
        overlay_base64 = image_to_base64(overlay_image)

        # Clean up uploaded file
        os.remove(filepath)

        return jsonify({
            'success': True,
            'original_image': original_base64,
            'mask_image': mask_base64,
            'overlay_image': overlay_base64
        })

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)