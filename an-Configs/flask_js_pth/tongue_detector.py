"""
舌象检测模块
基于现有的舌象分割模型实现
"""
import copy
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from PIL import Image

from config import Config
from utils.logger import get_logger

logger = get_logger(__name__)


def crop_center_image(source_img: PIL.Image.Image) -> Tuple[tuple, PIL.Image.Image]:
    """裁剪图像中心区域"""
    Width, Height = source_img.size
    left, upper = int(0.25 * Width), int(0.27 * Height)
    right, lower = int(Width - left), int(Height - upper + 0.14 * Height)
    crop_img: PIL.Image.Image = source_img.crop((left, upper, right, lower))
    return (left, upper, right, lower), crop_img


def find_connected_components(img: np.ndarray, min_size: int = 100) -> np.ndarray:
    """找到最大的连通域"""
    foreground_mask = np.any(img != [0, 0, 0], axis=2).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground_mask, connectivity=8)

    if num_labels <= 1 or len(stats) <= 1:
        return np.zeros_like(img)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    if stats[largest_label, cv2.CC_STAT_AREA] < min_size:
        return np.zeros_like(img)

    tongue_mask = (labels == largest_label).astype(np.uint8) * 255
    x, y, w, h, area = stats[largest_label]
    cropped_tongue = cv2.bitwise_and(img, img, mask=tongue_mask)
    return cropped_tongue


def bbox_img_func(img: np.ndarray, shift_v: int = 25) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """提取图像的边界框"""
    assert img is not None, '图片读取失败'
    max_img: np.ndarray = find_connected_components(img, min_size=100)
    mask: np.ndarray = np.any(max_img != [0, 0, 0], axis=2).astype(np.uint8)

    if np.sum(mask) == 0:
        return np.zeros_like(img), (0, 0, img.shape[0], img.shape[1])

    coords = cv2.findNonZero(mask)
    if coords is None:
        return np.zeros_like(img), (0, 0, img.shape[0], img.shape[1])

    x, y, w, h = cv2.boundingRect(coords)
    height, width, _ = img.shape
    shift_y = (height * 1.0 / width) * shift_v
    shift_x = min(shift_v, x)
    shift_y = min(shift_v, y)
    tongue = max_img[y - shift_y:y + h + shift_y, x - shift_x:x + w + shift_x]

    return tongue, (y - shift_y, x - shift_x, y + h + shift_y, x + w + shift_x)


class TongueDetector:
    """舌象检测器"""

    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        初始化舌象检测器

        Args:
            model_path: 舌象分割模型路径
            device: 计算设备
        """
        self.model_path = model_path or str(Config.TONGUE_MODEL_PATH)
        self.device = self._choose_device(device)
        self.net = None
        self._initialize_model()

    def _choose_device(self, device: str) -> torch.device:
        """选择计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 0:
                    device = 'cuda:0'
                else:
                    device = 'cpu'
            else:
                device = 'cpu'

        device_obj = torch.device(device)
        logger.info(f"舌象检测使用设备: {device_obj}")
        return device_obj

    def _initialize_model(self):
        """初始化模型"""
        try:
            logger.info(f"加载舌象分割模型: {self.model_path}")

            # 加载TorchScript模型
            self.net = torch.jit.load(self.model_path, map_location=self.device)
            self.net = self.net.to(self.device)
            self.net.eval()

            self.input_shape = Config.TONGUE_INPUT_SIZE

            logger.info("舌象分割模型加载成功")

        except Exception as e:
            logger.error(f"舌象分割模型加载失败: {str(e)}")
            raise

    def predict(self, image: PIL.Image.Image) -> Tuple[PIL.Image.Image, PIL.Image.Image]:
        """
        舌象分割预测

        Args:
            image: 输入图像

        Returns:
            (原图bbox结果, 无背景分割结果)
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')

        old_img: PIL.Image.Image = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # 预处理 - 使用训练时相同的归一化参数
        # 确保尺寸是32的倍数 (size_divisor=32)
        target_h, target_w = self.input_shape
        # 调整为32的倍数
        target_h = (target_h // 32) * 32
        target_w = (target_w // 32) * 32

        image_data = image.resize((target_w, target_h), Image.LANCZOS)
        temp = np.array(image_data, np.float32) / 255.0
        # 使用训练配置中的归一化参数
        mean = np.array([83.675, 156.28, 253.53], np.float32) / 255.0
        std = np.array([78.395, 87.12, 5.375], np.float32) / 255.0
        temp = (temp - mean) / std
        image_data = np.expand_dims(np.transpose(temp, (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).to(self.device)
            pr = self.net(images)[0]

        # 后处理
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)

        seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
        _pil_seg_image: Image.Image = Image.fromarray(np.uint8(seg_img))

        # 边界框裁剪
        seg_img_np = np.array(_pil_seg_image)
        cropped_img_np, (y_start, x_start, y_end, x_end) = bbox_img_func(seg_img_np, shift_v=1)
        PIL_bbox_no_background_image = Image.fromarray(cropped_img_np)
        source_seg_bbox_img: PIL.Image.Image = old_img.crop((x_start, y_start, x_end, y_end))

        return source_seg_bbox_img, PIL_bbox_no_background_image

    def detect_tongue_exists(self, input_image: PIL.Image.Image) -> Tuple[PIL.Image.Image, float]:
        """
        检测舌象是否存在并计算占比

        Args:
            input_image: 输入图像

        Returns:
            (舌象分割结果, 舌象占比百分比)
        """
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')

        # 中心裁剪
        crop_size, image = crop_center_image(input_image)
        cropped_old_img: PIL.Image.Image = copy.deepcopy(image)

        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]

        # 预处理 - 使用训练时相同的归一化参数
        # 确保尺寸是32的倍数 (size_divisor=32)
        target_h, target_w = self.input_shape
        # 调整为32的倍数
        target_h = (target_h // 32) * 32
        target_w = (target_w // 32) * 32

        image_data = image.resize((target_w, target_h), Image.LANCZOS)
        temp = np.array(image_data, np.float32) / 255.0
        # 使用训练配置中的归一化参数
        mean = np.array([83.675, 156.28, 253.53], np.float32) / 255.0
        std = np.array([78.395, 87.12, 5.375], np.float32) / 255.0
        temp = (temp - mean) / std
        image_data = np.expand_dims(np.transpose(temp, (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data).to(self.device)
            pr = self.net(images)[0]

        # 后处理
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)

        # 创建二值mask
        binary_mask = (np.expand_dims(pr != 0, -1) * 255).astype('uint8')
        binary_img: PIL.Image.Image = Image.fromarray(binary_mask.squeeze(-1))

        seg_img = (np.expand_dims(pr != 0, -1) * np.array(cropped_old_img, np.float32)).astype('uint8')

        # 恢复到原图尺寸
        source_seg_img = Image.new(input_image.mode, input_image.size)
        seg_img_PIL: Image.Image = Image.fromarray(seg_img)
        source_seg_img.paste(seg_img_PIL, crop_size)

        _pil_seg_image: Image.Image = source_seg_img

        # 边界框处理
        seg_img_np: np.ndarray = np.array(_pil_seg_image)
        cropped_img_np, crop_boxes = bbox_img_func(seg_img_np, shift_v=20)
        PIL_bbox_image: PIL.Image.Image = Image.fromarray(cropped_img_np)

        # 裁剪原图区域进行二次检测
        cropped_input_img: Image.Image = input_image.crop(
            (crop_boxes[1], crop_boxes[0], crop_boxes[3], crop_boxes[2]))
        source_seg_bbox_img, optimized_no_background_image = self.predict(cropped_input_img)

        # 计算舌象占比
        optimized_no_background_array: np.ndarray = np.array(optimized_no_background_image)
        tongue_mask: np.ndarray = np.any(optimized_no_background_array != [0, 0, 0], axis=2)
        tongue_pixels = np.sum(tongue_mask)
        tongue_ratio = tongue_pixels / (input_image.size[0] * input_image.size[1])

        return source_seg_bbox_img, tongue_ratio * 100

    def has_tongue(self, image: PIL.Image.Image, threshold: float = None) -> Tuple[bool, dict]:
        """
        检测图像中是否包含舌象

        Args:
            image: 输入图像
            threshold: 舌象占比阈值

        Returns:
            (has_tongue, detection_info): 是否有舌象和检测信息
        """
        try:
            if threshold is None:
                threshold = Config.TONGUE_RATIO_THRESH

            tongue_img, tongue_ratio = self.detect_tongue_exists(image)
            has_tongue = tongue_ratio >= threshold

            detection_info = {
                'tongue_ratio': float(round(tongue_ratio, 2)),
                'threshold': threshold,
                'has_tongue': has_tongue,
                'image_info': {
                    'width': image.width,
                    'height': image.height
                }
            }

            logger.info(f"舌象检测 - 占比: {tongue_ratio:.2f}%, 阈值: {threshold}%, 结果: {has_tongue}")

            return has_tongue, detection_info

        except Exception as e:
            logger.error(f"舌象检测失败: {str(e)}")
            return False, {'error': str(e)}

    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_type': 'tongue_segmentation',
            'model_path': str(self.model_path),
            'device': str(self.device),
            'input_size': self.input_shape,
            'threshold': Config.TONGUE_RATIO_THRESH
        }