# Preprocessing Techniques Guide

## Overview

This guide explains all **12+ preprocessing techniques** available for medical CT image processing.

---

## Preprocessing Techniques

### **Category 1: BASIC PROCESSING (Preparation)**

####  **RESIZING**
```
Purpose: Standardize image dimensions
Formula: Resize image to fixed size (e.g., 256×256)
Input: Variable-size image
Output: Fixed 256×256 image
```
**Why:** Neural networks require consistent input dimensions
**When:** Always use first
**Impact:** Enables batch processing

---

#### **MIN-MAX NORMALIZATION**
```
Formula: (x - min) / (max - min)
Range: [0, 1]
```
**Why:** Scales pixel values to standard range
**When:** Most supervised learning tasks
**Impact:** Prevents one feature from dominating

---

####  **Z-SCORE STANDARDIZATION**
```
Formula: (x - mean) / std
Range: approx [-3, 3]
```
**Why:** Centers data around 0 with standard deviation of 1
**When:** Deep learning models (especially batch normalization)
**Impact:** Faster convergence, better gradient flow

---

### **Category 2: CONTRAST ENHANCEMENT (Medical Imaging)**

####  **HISTOGRAM EQUALIZATION**
```
Purpose: Improve contrast distribution
Effect: Redistributes pixel values across full range
```
**Why:** Reveals hidden details in medical images
**When:** Low-contrast CT scans
**Impact:** Better visibility of anatomical structures

---

#### **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
```
Purpose: Adaptive histogram equalization
Region: Applied to local regions (not global)
Limit: Prevents over-amplification of noise
```
**Why:** Better than regular histogram equalization for medical images
**When:** When you need local contrast improvement
**Impact:** Preserves detail while enhancing contrast
**Parameters:**
- `clip_limit`: Controls contrast amplification (default: 2.0)
- `tile_size`: Size of local regions (default: 8×8)

---

### **Category 3: DENOISING (Noise Reduction)**

####  **GAUSSIAN BLUR**
```
Purpose: Smooth image, reduce noise
Kernel: Gaussian (bell-shaped)
Effect: Blurs edges and noise
```
**Why:** Removes high-frequency noise
**When:** Noisy CT images
**Impact:** Simplifies image, may lose fine details
**Trade-off:** Noise reduction vs detail preservation

---

####  **BILATERAL FILTER**
```
Purpose: Edge-preserving smoothing
Special: Blurs similar pixels while keeping edges sharp
```
**Why:** Removes noise without blurring edges
**When:** Want denoising but need to keep edges
**Impact:** Better than simple Gaussian blur for medical images
**Medical advantage:** Preserves tumor/lesion boundaries

---

####  **MORPHOLOGICAL OPERATIONS**
```
Types: Opening, Closing, Erosion, Dilation
Purpose: Remove small artifacts, fill holes
```
**Closing:** Removes small holes
**Opening:** Removes small objects
**When:** After denoising to clean up

---

### **Category 4: DATA AUGMENTATION (Training Data)**

####  **ROTATION**
```
Technique: Rotate image by angle (e.g., ±15°)
Purpose: Increase dataset diversity
```
**Why:** Model learn rotation-invariant features
**When:** Limited training data
**Impact:** Doubles/triples dataset without new images

---

#### **FLIPPING (Horizontal/Vertical)**
```
Types: Horizontal flip, Vertical flip
Purpose: Create mirror images
```
**Why:** Medical images have natural anatomical symmetry
**When:** Anatomically symmetric structures
**Impact:** Easy way to augment dataset

---

####  **ELASTIC DEFORMATION**
```
Purpose: Simulate anatomical variations
Effect: Slight warping of image
```
**Why:** Real medical images have natural deformations
**When:** Want to simulate patient variability
**Impact:** Models learn deformation-invariant features

---

####  **MULTI-SCALE ANALYSIS (Gaussian Pyramid)**
```
Purpose: Represent image at multiple scales
Levels: 1×, 2×, 4×, 8× downsampling
```
**Why:** Features exist at different scales
**When:** Need hierarchical feature extraction
**Impact:** Better captures both fine and coarse features

---

##  Recommended Pipelines

### **For General Medical Imaging:**
```
1. Resize → 256×256
2. Min-Max Normalize → [0, 1]
3. CLAHE → improve contrast
4. Bilateral Filter → denoise
5. Ready for training!
```

### **For Noisy Images:**
```
1. Resize
2. Bilateral Filter → aggressive denoising
3. Histogram Equalization → contrast
4. Z-score Standardization
5. Ready for training!
```

### **For Data Augmentation (Training):**
```
Base pipeline + any of:
- Rotation (±15°)
- Flipping (horizontal)
- Elastic Deformation
```

### **For Analysis:**
```
1. Resize
2. CLAHE → maximum detail
3. Multi-scale pyramid
4. Ready for feature extraction!
```

---

##  Preprocessing Impact on Data

| Technique | Input Range | Output Range | Use Case |
|-----------|------------|--------------|----------|
| Min-Max | Arbitrary | [0, 1] | General use |
| Z-score | Arbitrary | [-3, 3] | Deep learning |
| Histogram Eq | Arbitrary | [0, 1] | Low contrast |
| CLAHE | Arbitrary | [0, 1] | Medical images |
| Gaussian Blur | Arbitrary | Arbitrary | Denoising |
| Bilateral | Arbitrary | Arbitrary | Edge-preserving |

---

##  Quick Start

### Using the Preprocessor Class:

```python
from preprocessing_techniques import ImagePreprocessor

# Initialize
preprocessor = ImagePreprocessor()

# Define techniques
techniques = ['resize', 'normalize', 'clahe', 'bilateral']

# Process image
results = preprocessor.preprocess_pipeline(
    'image.jpg',
    techniques=techniques
)

# Access results
original = results['original']
normalized = results['normalized']
clahe_enhanced = results['clahe']
```

### Available Techniques:
- `resize` - Resize to 256×256
- `normalize` - Min-Max normalization
- `standardize` - Z-score standardization
- `histogram_eq` - Histogram equalization
- `clahe` - CLAHE enhancement
- `blur` - Gaussian blur
- `bilateral` - Bilateral filter
- `morphological` - Morphological operations

---

##  Key Parameters

### CLAHE
```python
clahe_enhancement(image, 
                 clip_limit=2.0,      # Contrast limit
                 tile_size=8)          # Region size
```

### Gaussian Blur
```python
gaussian_blur(image, 
             kernel_size=5)           # Blur strength
```

### Bilateral Filter
```python
bilateral_filter(image,
                diameter=9,            # Neighborhood size
                sigma_color=75,        # Color similarity
                sigma_space=75)        # Distance similarity
```

---

##  When to Use What

**Starting with noisy data?**
→ Use: Bilateral Filter + Histogram Equalization

**Working with deep learning?**
→ Use: Z-score Standardization

**Medical image analysis?**
→ Use: CLAHE + Edge-preserving filters

**Limited training data?**
→ Use: Augmentation (rotation, flipping, elastic deformation)

**Need to detect small structures?**
→ Use: CLAHE + Morphological Operations

---

##  Complete Example

```python
import cv2
import numpy as np
from preprocessing_techniques import ImagePreprocessor

# Load image
img = cv2.imread('ct_scan.jpg', cv2.IMREAD_GRAYSCALE)

# Create preprocessor
pp = ImagePreprocessor()

# Apply pipeline
results = pp.preprocess_pipeline(
    'ct_scan.jpg',
    techniques=['resize', 'normalize', 'clahe', 'bilateral']
)

# Use for training
medical_ready = results['bilateral']  # Use this for model
```

---

##  Performance Comparison

All techniques tested on sample data:

| Technique | Speed | Memory | Detail Preservation |
|-----------|-------|--------|-------------------|
| Resize | *** | Low | Full |
| Normalization | *** | Low | Full |
| Histogram Eq | ** | Low | High |
| CLAHE | * | Medium | Very High |
| Gaussian Blur | ** | Low | Medium |
| Bilateral | Moderate | Medium | High |

---

##  Resources

- OpenCV Docs: https://docs.opencv.org/
- Medical Image Processing: scikit-image, MedPy
- Data Augmentation: albumentations, imgaug

---

**Last Updated:** March 25, 2026
**Total Techniques:** 12+
**Categories:** 4
