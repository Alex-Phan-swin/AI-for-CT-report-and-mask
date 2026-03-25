# Transfer Learning Guide - Which Files to Use for Pre-trained Models

## Quick Answer

For **pre-trained models**, the next person should use these files:

| File | Purpose | Use |
|------|---------|-----|
| **dataset_metadata.csv**  | Train/Test split | Load which samples are for training vs testing |
| **preprocessed_data/**  | Image data | Load actual images for the model |
| **preprocessing_log.json** | Config info | Verify image size (256×256) and format (float32) |
| **dataset_config.json** | Dataset info | Reference dataset structure |

---

## File Details

### 1. **dataset_metadata.csv** (MOST IMPORTANT)

```csv
split,index,bdmap_id
train,0,BDMAP_00006697
train,1,BDMAP_00004878
test,0,BDMAP_00000005
...
```

**How to use:**
```python
import pandas as pd

df = pd.read_csv('dataset_metadata.csv')

# Get training samples
train_df = df[df['split'] == 'train']  # 13,032 samples

# Get test samples  
test_df = df[df['split'] == 'test']    # 5,492 samples

# Access BDMAP ID for image path
bdmap_id = train_df.iloc[0]['bdmap_id']  # BDMAP_00006697
```

---

### 2. **preprocessed_data/** (Image Folder)

**Location:** `./preprocessed_data/`

**Contains:** Preprocessed images (256×256, normalized)

**How to use:**
```python
import cv2

# Load image using BDMAP ID
bdmap_id = "BDMAP_00006697"
img_path = f"preprocessed_data/{bdmap_id}.jpg"
image = cv2.imread(img_path)

# Image properties:
# - Size: 256 × 256 pixels
# - Normalized: [0, 1]
# - Format: uint8 → float32
```

---

### 3. **preprocessing_log.json** (Config Reference)

```json
{
  "timestamp": "2026-03-25 23:34:15",
  "techniques_applied": ["resize", "normalize", "clahe"],
  "statistics": {
    "output_format": "float32",
    "output_size": "256x256",
    "techniques_count": 12
  }
}
```

**How to use:**
```python
import json

with open('preprocessing_log.json') as f:
    config = json.load(f)

print(config['statistics']['output_size'])     # "256x256"
print(config['statistics']['output_format'])   # "float32"
```

---

### 4. **dataset_config.json** (Info Only)

**Purpose:** Reference only - shows dataset structure

```json
{
  "dataset_name": "AbdomenAtlas3.0Mini",
  "total_samples": 18524,
  "splits": {
    "train": {"samples": 13032, "features": ["BDMAP ID"]},
    "test": {"samples": 5492, "features": ["BDMAP ID"]}
  }
}
```

---

##  Pre-trained Models to Use

### **RECOMMENDED: ResNet50** 
```python
import torchvision.models as models
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
```
- Fast & accurate
- Perfect for medical imaging
- 224×224 input (resize from 256×256)

### **ALSO GOOD: DenseNet121**
```python
model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
```
- Better for medical images
- Efficient gradient flow
- Good accuracy

### **LIGHTWEIGHT: EfficientNet-B0**
```python
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
```
- Mobile-friendly
- Fast inference
- Good accuracy-speed trade-off

---

## 🔧 Complete Example Code

```python
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# ========== STEP 1: Load Metadata ==========
df = pd.read_csv('dataset_metadata.csv')
train_df = df[df['split'] == 'train']  # 13,032 samples
test_df = df[df['split'] == 'test']    # 5,492 samples

print(f"Training: {len(train_df)} | Test: {len(test_df)}")

# ========== STEP 2: Create Dataset ==========
class MedicalImageDataset(Dataset):
    def __init__(self, dataframe, img_dir='preprocessed_data'):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get BDMAP ID from metadata
        bdmap_id = self.df.iloc[idx]['bdmap_id']
        
        # Load image
        img_path = f"{self.img_dir}/{bdmap_id}.jpg"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        return image

# ========== STEP 3: Create DataLoaders ==========
train_dataset = MedicalImageDataset(train_df)
test_dataset = MedicalImageDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ========== STEP 4: Load Pre-trained Model ==========
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Modify for grayscale (1 input channel instead of 3)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify output layer for your task (example: 2 classes)
num_classes = 2
model.fc = nn.Linear(2048, num_classes)

# ========== STEP 5: Setup Training ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ========== STEP 6: Training Loop ==========
for epoch in range(10):
    model.train()
    for images in train_loader:
        images = images.to(device)
        outputs = model(images)
        # ... training code
        
    print(f"Epoch {epoch+1}/10 complete")

print("✓ Training complete!")
```

---

## Checklist Before Starting

- [ ] `dataset_metadata.csv` exists
- [ ] `preprocessed_data/` folder with images exists
- [ ] `preprocessing_log.json` shows 256×256 images
- [ ] Python packages installed: `torch`, `torchvision`, `opencv-python`, `pandas`
- [ ] GPU available (optional but recommended)
- [ ] You have BDMAP IDs from dataset_metadata.csv
- [ ] Image path formula: `preprocessed_data/{bdmap_id}.jpg`

---

## Workflow Summary

```
1. Load dataset_metadata.csv
   ↓
2. Split into train/test using 'split' column
   ↓
3. Use BDMAP IDs to load images from preprocessed_data/
   ↓
4. Create PyTorch Dataset & DataLoader
   ↓
5. Load pre-trained ResNet50 (or other model)
   ↓
6. Modify input layer for grayscale images
   ↓
7. Modify output layer for your classification task
   ↓
8. Fine-tune on your medical imaging data
   ↓
9. Evaluate on test set
```

---

##  Key Points

- **dataset_metadata.csv** = Train/test labels + BDMAP IDs
- **preprocessed_data/** = The actual image files
- **preprocessing_log.json** = Verification that images are 256×256
- **Pre-trained weights** = Download from torchvision (automatic)

**For the next person:**
1. Use `dataset_metadata.csv` to know which samples to train on
2. Use `preprocessed_data/` folder to load images
3. Use BDMAP ID as image filename: `{bdmap_id}.jpg`
4. Load any pre-trained model from torchvision
5. Modify for grayscale input & your output classes
6. Fine-tune on the training samples

---

**Last Updated:** March 25, 2026
**Use These Files:** dataset_metadata.csv + preprocessed_data/
**Recommended Model:** ResNet50
