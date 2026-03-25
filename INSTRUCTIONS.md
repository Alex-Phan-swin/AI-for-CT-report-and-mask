#  AbdomenAtlas3.0Mini - Data Extraction Pipeline

Complete Python scripts for downloading and processing the AbdomenAtlas3.0Mini medical imaging dataset.

## 📋 **Quick Start**

### Installation
```bash
pip install datasets huggingface-hub pandas numpy tqdm opencv-python scikit-learn
```

### Run Everything at Once
```bash
python run_all.py
```

### Or Run Individual Steps

**Step 1: Download Dataset**
```bash
python download_dataset.py
```

**Step 2: Extract & Process Data**
```bash
python extract_process_images.py
```

**Step 3: Preprocess with Advanced Techniques** ⭐
```bash
python preprocessing_techniques.py
```

---

## 📁 **File Structure**

```
project/
├── download_dataset.py              # Download from Hugging Face
├── extract_process_images.py        # Extract and analyze data
├── preprocessing_techniques.py       # 12+ preprocessing techniques ⭐
├── run_all.py                       # Master script (runs everything)
├── dataset_cache/                   # Downloaded dataset (auto-created)
├── preprocessed_data/               # Preprocessed images (auto-created)
├── dataset_metadata.csv             # Extracted metadata (auto-generated)
├── dataset_config.json              # Dataset config (auto-generated)
├── preprocessing_log.json           # Preprocessing tracking (auto-generated)
├── INSTRUCTIONS.md                  # This file
├── PREPROCESSING_GUIDE.md           # Detailed preprocessing guide ⭐
└── README.md                        # Original README
```

---

## 📋 **Script Details**

### 1. **download_dataset.py**
Downloads AbdomenAtlas3.0Mini from Hugging Face

**What it does:**
-  Downloads dataset metadata
-  Saves to local cache
-  Displays dataset structure
-  Shows sample data

**Output:**
- `dataset_cache/` folder with cached data

**Usage:**
```bash
python download_dataset.py
```

---

### 2. **extract_process_images.py**
Extracts and analyzes the dataset

**What it does:**
-  Loads cached dataset
-  Extracts BDMAP IDs and metadata
-  Analyzes dataset statistics
-  Saves metadata to CSV
-  Saves configuration to JSON

**Output:**
- `dataset_metadata.csv` - All samples with BDMAP IDs
- `dataset_config.json` - Dataset structure info

**Usage:**
```bash
python extract_process_images.py
```

---

### 3. **run_all.py**
Master script that runs everything in order

**What it does:**
-  Runs download_dataset.py
-  Runs extract_process_images.py
-  Tracks progress
-  Shows final summary

**Usage:**
```bash
python run_all.py
```

---
### 4. **preprocessing_techniques.py** ⭐ NEW
Comprehensive image preprocessing with 12+ techniques

**What it does:**
- ✅ Implements 12+ preprocessing techniques
- ✅ Demonstrates all techniques with examples
- ✅ Provides preprocessing class for custom pipelines
- ✅ Generates preprocessing logs
- ✅ Shows input/output ranges and statistics

**Output:**
- `preprocessed_data/` folder with processed images
- `preprocessing_log.json` - Tracking applied techniques

**Usage:**
```bash
python preprocessing_techniques.py
```

---

## 🔬 Preprocessing Techniques (12+)

### **Category 1: BASIC PROCESSING**

#### 1. **Resizing**
- **Purpose:** Standardize image dimensions
- **Output Size:** 256×256
- **Why Use:** Neural networks require consistent input
- **Impact:** Enables batch processing

#### 2. **Min-Max Normalization**
- **Formula:** (x - min) / (max - min)
- **Output Range:** [0, 1]
- **Why Use:** Scales pixel values to standard range
- **Best For:** Most supervised learning tasks

#### 3. **Z-Score Standardization**
- **Formula:** (x - mean) / std
- **Output Range:** [-3, 3]
- **Why Use:** Centers data around 0 with std=1
- **Best For:** Deep learning models with batch normalization

---

### **Category 2: CONTRAST ENHANCEMENT**

#### 4. **Histogram Equalization**
- **Purpose:** Improve contrast distribution
- **Method:** Redistributes pixel values across full range
- **Why Use:** Reveals hidden details in medical images
- **Best For:** Low-contrast CT scans

#### 5. **CLAHE (Contrast Limited Adaptive Histogram Equalization)** ⭐ RECOMMENDED
- **Purpose:** Adaptive histogram equalization
- **Method:** Applied to local regions (not global)
- **Advantage:** Better than regular histogram equalization
- **Best For:** Medical imaging - preserves details while enhancing contrast
- **Parameters:**
  - `clip_limit`: 2.0 (controls contrast amplification)
  - `tile_size`: 8×8 (size of local regions)

---

### **Category 3: DENOISING**

#### 6. **Gaussian Blur**
- **Purpose:** Smooth image and reduce noise
- **Method:** Gaussian smoothing filter
- **Why Use:** Removes high-frequency noise
- **Trade-off:** Noise reduction vs detail preservation

#### 7. **Bilateral Filter** ⭐ RECOMMENDED
- **Purpose:** Edge-preserving smoothing
- **Advantage:** Blurs noise while keeping edges sharp
- **Why Use:** Better than Gaussian blur for medical images
- **Medical Advantage:** Preserves tumor/lesion boundaries

#### 8. **Morphological Operations**
- **Types:** Opening (remove small objects), Closing (remove holes)
- **Purpose:** Clean up image artifacts
- **Why Use:** Remove small noise after denoising

---

### **Category 4: DATA AUGMENTATION**

#### 9. **Rotation**
- **Purpose:** Increase dataset diversity
- **Angles:** ±15° (configurable)
- **Why Use:** Model learns rotation-invariant features
- **When Limited Data:** Doubles dataset without new images

#### 10. **Flipping** 
- **Types:** Horizontal flip, Vertical flip
- **Purpose:** Create mirror images
- **Why Use:** Medical images have anatomical symmetry
- **Best For:** Symmetric anatomical structures

#### 11. **Elastic Deformation**
- **Purpose:** Simulate anatomical variations
- **Effect:** Slight warping of image
- **Why Use:** Real medical images have natural deformations
- **Benefit:** Models learn deformation-invariant features

#### 12. **Gaussian Pyramid**
- **Purpose:** Multi-scale image representation
- **Levels:** 1×, 2×, 4×, 8× downsampling
- **Why Use:** Features exist at different scales
- **Benefit:** Better captures fine and coarse features

---

## 📊 Recommended Preprocessing Pipelines

### **Pipeline 1: General Medical Imaging (STANDARD)**
```
1. Resize → 256×256
2. Min-Max Normalize → [0, 1]
3. CLAHE → improve contrast
4. Bilateral Filter → denoise
5. Ready for training!
```

### **Pipeline 2: Noisy Images (AGGRESSIVE DENOISING)**
```
1. Resize → 256×256
2. Bilateral Filter → aggressive denoising
3. Histogram Equalization → contrast
4. Z-score Standardization
5. Ready for training!
```

### **Pipeline 3: Training with Augmentation**
Base pipeline + choose any:
- Rotation (±15°)
- Flipping (horizontal/vertical)
- Elastic Deformation

### **Pipeline 4: Analysis and Feature Extraction**
```
1. Resize
2. CLAHE → maximum detail
3. Multi-scale Pyramid
4. Ready for analysis!
```

---

## 📈 Preprocessing Impact Summary

| Technique | Input | Output | Use Case |
|-----------|-------|--------|----------|
| Min-Max | Any | [0, 1] | General use |
| Z-score | Any | [-3, 3] | Deep learning |
| Histogram Eq | Any | [0, 1] | Low contrast |
| **CLAHE** | Any | [0, 1] | **Medical images** ⭐ |
| Gaussian Blur | Any | Any | Denoising |
| Bilateral | Any | Any | Edge-preserving |
| Rotation | Any | Any | Data augmentation |
| Elastic Deform | Any | Any | Augmentation |

---

## 💡 Quick Decision Guide

| Your Situation | Recommended Technique |
|---|---|
| Just starting | Use Pipeline 1 (Standard) |
| Noisy images | Add Bilateral Filter |
| Low contrast | Use CLAHE enhancement |
| Limited training data | Add augmentation (rotation, flipping) |
| Need edge details | Use Bilateral Filter |
| Multiple scales | Use Gaussian Pyramid |
| Medical CT scans | Use CLAHE + Bilateral ⭐ |

---

## 🔧 Using Preprocessing in Your Code

```python
from preprocessing_techniques import ImagePreprocessor

# Initialize preprocessor
preprocessor = ImagePreprocessor()

# Define techniques to apply
techniques = ['resize', 'normalize', 'clahe', 'bilateral']

# Process single image
results = preprocessor.preprocess_pipeline(
    'image.jpg',
    techniques=techniques
)

# Access different versions
original = results['original']
normalized = results['normalized']
clahe = results['clahe']
final = results['bilateral']  # Use for training
```

---

## 📝 Preprocessing Output Files

When you run `preprocessing_techniques.py`:

**File:** `preprocessing_log.json`
```json
{
  "timestamp": "2026-03-25 23:34:15",
  "techniques_applied": [
    "resize",
    "normalize", 
    "clahe",
    "bilateral"
  ],
  "statistics": {
    "total_images_processed": 0,
    "output_format": "float32",
    "output_size": "256x256",
    "techniques_count": 12
  }
}
```

---

## 📊 **Dataset Information**

**Dataset Name:** AbdomenAtlas3.0Mini

**Total Samples:** 18,524
- Training: 13,032 samples
- Testing: 5,492 samples

**Features:**
- BDMAP ID (unique identifier)

---

## 📋 **Output Examples**

### dataset_metadata.csv
```
split,index,bdmap_id
train,0,001-0001
train,1,001-0002
test,0,002-0001
...
```

### dataset_config.json
```json
{
  "dataset_name": "AbdomenAtlas3.0Mini",
  "total_samples": 18524,
  "splits": {
    "train": {
      "samples": 13032,
      "features": ["BDMAP ID"]
    },
    "test": {
      "samples": 5492,
      "features": ["BDMAP ID"]
    }
  }
}
```

---

## ⚠️ **Common Issues & Solutions**

### Issue: Module not found
```bash
pip install datasets huggingface-hub pandas numpy tqdm
```

### Issue: Permission denied on Hugging Face
```bash
huggingface-cli login
# Paste your token from: https://huggingface.co/settings/tokens
```

### Issue: Cache already exists
Scripts automatically handle existing cache - just re-run!

---

##  Next Steps

After running the scripts:

1. **Review extracted data:**
   ```bash
   head dataset_metadata.csv
   cat dataset_config.json
   cat preprocessing_log.json
   ```

2. **Preprocess your images:**
   ```bash
   python preprocessing_techniques.py
   ```
   This will:
   - ✅ Demonstrate all 12+ techniques
   - ✅ Generate preprocessing logs
   - ✅ Show input/output statistics
   - ✅ Create output directory for processed images

3. **Use preprocessed data in your model:**
   ```python
   from preprocessing_techniques import ImagePreprocessor
   
   preprocessor = ImagePreprocessor()
   techniques = ['resize', 'normalize', 'clahe', 'bilateral']
   results = preprocessor.preprocess_pipeline('image.jpg', techniques)
   ```

4. **For detailed preprocessing guide:**
   - Open `PREPROCESSING_GUIDE.md` for comprehensive documentation
   - Choose which techniques to use based on your needs
   - Follow recommended pipelines for your use case

5. **Advanced usage with dataset:**
   ```python
   from datasets import load_dataset
   from preprocessing_techniques import ImagePreprocessor
   
   # Load dataset
   ds = load_dataset(
       "AbdomenAtlas/AbdomenAtlas3.0Mini",
       cache_dir="./dataset_cache"
   )
   
   # Preprocess
   preprocessor = ImagePreprocessor()
   processed = preprocessor.preprocess_pipeline(
       image_path, 
       techniques=['resize', 'normalize', 'clahe']
   )
   ```

---

##  Complete Workflow

```
Step 1: python download_dataset.py
        ↓
Step 2: python extract_process_images.py
        ↓
Step 3: python preprocessing_techniques.py
        ↓
Step 4: Build your model with preprocessed images
        ↓
Step 5: Train and evaluate
```

Or run all at once:
```bash
python run_all.py  # Steps 1-2
python preprocessing_techniques.py  # Step 3
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `INSTRUCTIONS.md` | This file - complete setup guide |
| `PREPROCESSING_GUIDE.md` | Detailed guide for all 12+ techniques |
| `preprocessing_log.json` | Tracks preprocessing configuration |

---

## Notes

- ✅ Dataset contains BDMAP IDs (metadata)
- ✅ Preprocessing has 12+ techniques implemented
- ✅ CLAHE recommended for medical CT images
- ✅ Bilateral Filter best for edge-preserving denoising
- ✅ All processing is done locally after first download
- ℹ️ Actual CT images from external sources (use BDMAP IDs to reference)

---

## Support

For more information:
- 📖 [Hugging Face Datasets](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini)
- 📖 [Hugging Face Documentation](https://huggingface.co/docs/datasets)
- 📖 [OpenCV Documentation](https://docs.opencv.org/)
- 📖 [Scikit-Learn Documentation](https://scikit-learn.org/)

---

**Last Updated:** March 25, 2026
**Version:** 2.1 (with Preprocessing)
**Total Techniques:** 12+
**Recommended for:** Medical CT Image Analysis
