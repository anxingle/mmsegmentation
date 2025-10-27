#!/usr/bin/env python3
import os
import cv2
import numpy as np
from pathlib import Path

def generate_zero_masks():
    """
    Generate zero-filled mask images for all images in img_dir
    """
    img_dir = Path("datasets/background_images/img_dir")
    ann_dir = Path("datasets/background_images/ann_dir")

    # Ensure directories exist
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []

    for ext in image_extensions:
        image_files.extend(img_dir.glob(f"*{ext}"))
        image_files.extend(img_dir.glob(f"*{ext.upper()}"))

    print(f"Found {len(image_files)} images in {img_dir}")

    generated_count = 0

    for img_path in image_files:
        try:
            # Read the original image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not read {img_path}")
                continue

            height, width = img.shape[:2]

            # Create zero mask (same size, single channel, all zeros)
            mask = np.zeros((height, width), dtype=np.uint8)

            # Create corresponding mask file path
            mask_path = ann_dir / img_path.name

            # Save the mask
            cv2.imwrite(str(mask_path), mask)

            generated_count += 1

            if generated_count % 20 == 0:
                print(f"Generated {generated_count} masks...")

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Successfully generated {generated_count} zero-filled masks in {ann_dir}")
    print(f"Mask directory: {ann_dir.absolute()}")

if __name__ == "__main__":
    generate_zero_masks()