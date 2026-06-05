import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

# =========================================================
# PATHS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "images_dataset"
OUTPUT_DIR = BASE_DIR / "output"

CLASSES = ["Healthy", "Hepatic_Steatosis"]
TARGET_SIZE = (224, 224)

# =========================================================
# TECHNIQUE 1: RESIZING
# =========================================================
def resize_image(image: Image.Image, target_size=(224, 224)) -> Image.Image:
    return image.resize(target_size)

# =========================================================
# TECHNIQUE 2: NORMALIZATION
# =========================================================
def normalize_image(image: Image.Image) -> np.ndarray:
    return np.array(image).astype(np.float32) / 255.0

# =========================================================
# TECHNIQUE 3: HORIZONTAL FLIP
# =========================================================
def horizontal_flip(image: Image.Image) -> Image.Image:
    return ImageOps.mirror(image)

# =========================================================
# TECHNIQUE 4: ROTATION
# =========================================================
def rotate_image(image: Image.Image) -> Image.Image:
    angle = random.choice([90, 180, 270])
    return image.rotate(angle)

# =========================================================
# TECHNIQUE 5: BRIGHTNESS ADJUSTMENT
# =========================================================
def brightness_adjust(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(random.uniform(0.8, 1.2))

# =========================================================
# TECHNIQUE 6: CONTRAST ADJUSTMENT
# =========================================================
def contrast_adjust(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(random.uniform(0.8, 1.2))

# =========================================================
# TECHNIQUE 7: GAUSSIAN BLUR / DENOISING
# =========================================================
def gaussian_blur(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    blurred = cv2.GaussianBlur(arr, (5, 5), 0)
    return Image.fromarray(blurred)

# =========================================================
# TECHNIQUE 8: CLAHE
# =========================================================
def clahe_enhancement(image: Image.Image) -> Image.Image:
    arr = np.array(image)

    if len(arr.shape) == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif len(arr.shape) == 2:
        gray = arr
    else:
        return image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return Image.fromarray(enhanced)

# =========================================================
# TECHNIQUE 9: HISTOGRAM EQUALIZATION
# =========================================================
def histogram_equalization(image: Image.Image) -> Image.Image:
    arr = np.array(image)

    if len(arr.shape) == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif len(arr.shape) == 2:
        gray = arr
    else:
        return image

    equalized = cv2.equalizeHist(gray)
    return Image.fromarray(equalized)

# =========================================================
# TECHNIQUE 10: EDGE DETECTION
# =========================================================
def edge_detection(image: Image.Image) -> Image.Image:
    arr = np.array(image)

    if len(arr.shape) == 3 and arr.shape[2] == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    elif len(arr.shape) == 2:
        gray = arr
    else:
        return image

    edges = cv2.Canny(gray, 100, 200)
    return Image.fromarray(edges)

# =========================================================
# TECHNIQUE 11: PADDING
# =========================================================
def pad_image(image: Image.Image, target_size=(224, 224)) -> Image.Image:
    return ImageOps.pad(image, target_size, color=0)

# =========================================================
# TECHNIQUE 12: RANDOM ZOOM
# =========================================================
def random_zoom(image: Image.Image) -> Image.Image:
    w, h = image.size
    zoom_factor = random.uniform(1.0, 1.15)

    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)

    zoomed = image.resize((new_w, new_h))

    left = (new_w - w) // 2
    top = (new_h - h) // 2
    right = left + w
    bottom = top + h

    return zoomed.crop((left, top, right, bottom))

# =========================================================
# TECHNIQUE 13: RANDOM SHIFT / TRANSLATION
# =========================================================
def random_shift(image: Image.Image) -> Image.Image:
    arr = np.array(image)
    rows, cols = arr.shape[:2]

    tx = random.randint(-10, 10)
    ty = random.randint(-10, 10)

    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(arr, matrix, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    return Image.fromarray(shifted)

# =========================================================
# TECHNIQUE 14: INVERSION
# =========================================================
def invert_image(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return ImageOps.invert(image)

# =========================================================
# TECHNIQUE 15: GAUSSIAN NOISE
# =========================================================
def add_noise(image: Image.Image) -> Image.Image:
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, 10, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

# =========================================================
# SAVE NPY
# =========================================================
def save_npy(image: Image.Image, output_path_without_extension: Path) -> None:
    npy_array = normalize_image(image)
    np.save(str(output_path_without_extension) + ".npy", npy_array)

# =========================================================
# MAIN PREPROCESSING PIPELINE
# =========================================================
def preprocess_single_image(image_path: Path, output_png_path: Path, output_npy_base: Path) -> None:
    image = Image.open(image_path).convert("RGB")

    # Base preprocessing
    image = resize_image(image, TARGET_SIZE)
    image = pad_image(image, TARGET_SIZE)

    # Apply enhancement / augmentation techniques
    image = brightness_adjust(image)
    image = contrast_adjust(image)

    if random.random() > 0.5:
        image = horizontal_flip(image)

    if random.random() > 0.5:
        image = rotate_image(image)

    if random.random() > 0.5:
        image = random_zoom(image)

    if random.random() > 0.5:
        image = random_shift(image)

    if random.random() > 0.7:
        image = gaussian_blur(image)

    if random.random() > 0.7:
        image = add_noise(image)

    # Extra feature enhancement operations
    # Applied only to create better processed training-style image variants
    if random.random() > 0.7:
        image = clahe_enhancement(image)

    if random.random() > 0.8:
        image = histogram_equalization(image)

    if random.random() > 0.9:
        image = edge_detection(image)

    if random.random() > 0.95:
        image = invert_image(image)

    # Final save
    image.save(output_png_path)
    save_npy(image, output_npy_base)

# =========================================================
# PROCESS FULL DATASET
# =========================================================
def process_dataset() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    for class_name in CLASSES:
        class_input_dir = INPUT_DIR / class_name
        class_output_dir = OUTPUT_DIR / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        if not class_input_dir.exists():
            print(f"Folder not found: {class_input_dir}")
            continue

        image_files = sorted([
            f for f in class_input_dir.iterdir()
            if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])

        print(f"\nProcessing class: {class_name}")
        print(f"Total images found: {len(image_files)}")

        for idx, image_path in enumerate(image_files, start=1):
            output_png_path = class_output_dir / f"img-{idx}.png"
            output_npy_base = class_output_dir / f"img-{idx}"

            try:
                preprocess_single_image(image_path, output_png_path, output_npy_base)
                print(f"Saved: {output_png_path.name}")
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    process_dataset()