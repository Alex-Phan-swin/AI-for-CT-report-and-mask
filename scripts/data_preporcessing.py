import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2

# Directories
input_dir = 'images_dataset'  # Folder containing your images
output_dir = 'output'         # Folder to save preprocessed images

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to preprocess image (resize and normalize)
def preprocess_image(image_path, target_size=(224, 224)):
    """
    Resize the image and normalize pixel values to [0, 1]
    """
    img = Image.open(image_path)  # Open image
    img_resized = img.resize(target_size)  # Resize image
    img_array = np.array(img_resized) / 255.0  # Normalize pixel values
    return img_resized, img_array

# Data Augmentation Function
def augment_image(image):
    """
    Apply data augmentation techniques like flip, rotation, and brightness change
    """
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally
    
    if random.random() > 0.5:
        image = image.rotate(random.choice([90, 180, 270]))  # Random rotation
    
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.7, 1.3))  # Random brightness change
    
    return image

# Denoising (Gaussian blur)
def denoise_image(image):
    """
    Apply Gaussian blur to reduce image noise
    """
    img_array = np.array(image)  # Convert to numpy array
    img_denoised = cv2.GaussianBlur(img_array, (5, 5), 0)  # Apply Gaussian blur
    return Image.fromarray(img_denoised)  # Convert back to image

# Contrast Enhancement (CLAHE)
def enhance_contrast(image):
    """
    Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE) to enhance contrast.
    The image is first converted to RGB (if necessary) and then to grayscale before applying CLAHE.
    """
    img_array = np.array(image)  # Convert to numpy array
    
    # Ensure the image is in RGB format
    if len(img_array.shape) == 3:  # Check if the image has multiple channels
        if img_array.shape[2] == 4:  # Check for RGBA format
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)  # Convert RGBA to RGB
        elif img_array.shape[2] == 3:  # Check for RGB format
            pass  # Already in RGB format, no need to convert
        else:
            raise ValueError("Unexpected number of channels in image: expected 3 (RGB) or 4 (RGBA).")
    elif len(img_array.shape) == 2:  # Grayscale image (single channel)
        img_gray = img_array
    else:
        raise ValueError("Unexpected image dimensions: expected 2 or 3 dimensions.")
    
    # Convert to grayscale if image is in RGB or RGBA
    if len(img_array.shape) == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    
    # Create CLAHE object with specified parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Create CLAHE object
    img_enhanced = clahe.apply(img_gray)  # Apply CLAHE
    
    # Convert back to a PIL image
    img_enhanced = Image.fromarray(img_enhanced)
    
    return img_enhanced   
# Histogram Equalization
def equalize_histogram(image):
    """
    Apply histogram equalization to improve the overall contrast of the image.
    """
    img_array = np.array(image)  # Convert to numpy array
    img_eq = cv2.equalizeHist(img_array)  # Apply histogram equalization
    return Image.fromarray(img_eq)  # Convert back to image

# Edge Detection (Canny)
def edge_detection(image):
    """
    Apply edge detection (Canny) to highlight boundaries of structures in the image.
    """
    img_array = np.array(image)
    edges = cv2.Canny(img_array, 100, 200)  # Apply Canny edge detection
    return Image.fromarray(edges)  # Convert back to image

# Padding (To Maintain Aspect Ratio)
def pad_image(image, target_size=(224, 224)):
    """
    Apply padding to maintain aspect ratio when resizing.
    """
    img_resized = image.resize(target_size)  # Resize image
    img_padded = Image.new('RGB', target_size, (0, 0, 0))  # Create black padding
    img_padded.paste(img_resized, (0, 0))  # Paste resized image on top of padding
    return img_padded

# Random Zoom
def random_zoom(image):
    """
    Randomly zoom into the image to simulate different focal lengths.
    """
    zoom_factor = random.uniform(1.0, 1.2)  # Random zoom factor between 1.0 and 1.2
    img_zoomed = image.resize((int(image.width * zoom_factor), int(image.height * zoom_factor)))
    img_zoomed = img_zoomed.resize((224, 224))  # Resize back to target size
    return img_zoomed

# Rotation
def rotate_image(image):
    """
    Apply random rotation (90, 180, or 270 degrees) to the image.
    """
    return image.rotate(random.choice([90, 180, 270]))  # Random rotation

# Random Shifting (Translation)
def shift_image(image):
    """
    Randomly shift the image along x and y axes to simulate slight movements.
    """
    width, height = image.size
    x_translation = random.randint(-10, 10)
    y_translation = random.randint(-10, 10)
    img_translated = image.transform((width, height), Image.AFFINE, (1, 0, x_translation, 0, 1, y_translation))
    return img_translated

# Inversion (Negative Imaging)
def invert_image(image):
    """
    Invert the image to simulate contrast inversion.
    """
    return Image.eval(image, lambda x: 255 - x)  # Invert the image pixel values

# Adding Gaussian Noise
def add_noise(image):
    """
    Add Gaussian noise to the image to simulate noisy medical images.
    """
    img_array = np.array(image)
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, img_array.shape)
    noisy_img = img_array + gauss
    noisy_img = np.clip(noisy_img, 0, 255)  # Clip the values to stay within valid range
    return Image.fromarray(noisy_img.astype('uint8'))

# Function to process and save images
def process_images(input_dir, output_dir, target_size=(224, 224), augment=True, denoise=True, enhance=True):
    """
    Process all images in the input directory, apply the techniques, and save them
    """
    processed_count = 0
    
    # Loop through all files in the input directory
    for idx, filename in enumerate(os.listdir(input_dir), start=1):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Only process image files
            image_path = os.path.join(input_dir, filename)
            
            # Preprocess the image (resize and normalize)
            img_resized, img_array = preprocess_image(image_path, target_size)
            
            # Apply data augmentation if required
            if augment:
                img_resized = augment_image(img_resized)  # Data Augmentation

            # Apply denoising if required
            if denoise:
                img_resized = denoise_image(img_resized)  # Denoising

            # Apply contrast enhancement if required
            if enhance:
                img_resized = enhance_contrast(img_resized)  # Contrast Enhancement

            # Apply histogram equalization
            img_resized = equalize_histogram(img_resized)

            # Apply edge detection (Canny)
            img_resized = edge_detection(img_resized)

            # Apply padding to maintain aspect ratio
            img_resized = pad_image(img_resized)

            # Apply random zoom
            img_resized = random_zoom(img_resized)

            # Apply random rotation
            img_resized = rotate_image(img_resized)

            # Apply random shifting (translation)
            img_resized = shift_image(img_resized)

            # Apply inversion (negative imaging)
            img_resized = invert_image(img_resized)

            # Apply Gaussian noise (optional)
            img_resized = add_noise(img_resized)

            # Save the processed image
            output_image_path = os.path.join(output_dir, f"img-{idx}.png")
            img_resized.save(output_image_path)  # Save resized image
            
            # Optionally, save the numpy array as .npy file
            np.save(output_image_path.replace('.png', '.npy'), img_array)
            
            processed_count += 1
            print(f"Processed and saved: img-{idx}.png")
    
    print(f"Data preprocessing complete! Processed {processed_count} images.")

# Run preprocessing
process_images(input_dir, output_dir, target_size=(224, 224), augment=True, denoise=True, enhance=True)