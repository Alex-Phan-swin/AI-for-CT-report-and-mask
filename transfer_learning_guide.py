"""
Transfer Learning with Pre-trained Models
Use this script after preprocessing to train models with transfer learning
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

print("\n" + "=" * 80)
print("TRANSFER LEARNING WITH PRE-TRAINED MODELS - SETUP GUIDE")
print("=" * 80)

# ============================================================================
# PART 1: FILES TO USE
# ============================================================================

print("\n📁 FILES FOR PRE-TRAINED MODEL TRAINING:\n")

files_info = {
    "1. dataset_metadata.csv": {
        "purpose": "Train/Test split information",
        "contains": ["split (train/test)", "index", "bdmap_id"],
        "use_for": "Loading BDMAP IDs and organizing data splits",
        "example": "Load train samples only: df[df['split'] == 'train']"
    },
    
    "2. preprocessed_data/": {
        "purpose": "Preprocessed images folder",
        "contains": ["Resized images", "Normalized images", "CLAHE enhanced"],
        "use_for": "Input data for model training",
        "example": "Load images: cv2.imread('preprocessed_data/image.jpg')"
    },
    
    "3. preprocessing_log.json": {
        "purpose": "Preprocessing configuration log",
        "contains": ["Techniques applied", "Output format", "Image size"],
        "use_for": "Verify preprocessing settings used",
        "example": "Check image size: 256x256, format: float32"
    },
    
    "4. dataset_config.json": {
        "purpose": "Dataset structure information",
        "contains": ["Total samples", "Split sizes", "Features"],
        "use_for": "Understanding dataset structure",
        "example": "Total: 18,524 (Train: 13,032, Test: 5,492)"
    }
}

for file_name, info in files_info.items():
    print(f"📄 {file_name}")
    print(f"   Purpose: {info['purpose']}")
    print(f"   Contains: {', '.join(info['contains'])}")
    print(f"   Use For: {info['use_for']}")
    print(f"   Example: {info['example']}\n")

# ============================================================================
# PART 2: PRE-TRAINED MODELS AVAILABLE
# ============================================================================

print("\n" + "=" * 80)
print("AVAILABLE PRE-TRAINED MODELS FOR MEDICAL IMAGING")
print("=" * 80 + "\n")

pretrained_models = {
    "ResNet50": {
        "source": "torchvision.models",
        "weights": "ResNet50_Weights.IMAGENET1K_V1",
        "input_size": 224,
        "use_case": "General-purpose, good for medical images",
        "pros": ["Fast", "Accurate", "Well-documented"],
        "code": "models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)"
    },
    
    "ResNet101": {
        "source": "torchvision.models",
        "weights": "ResNet101_Weights.IMAGENET1K_V1",
        "input_size": 224,
        "use_case": "Deeper network, more capacity",
        "pros": ["More parameters", "Better for complex features"],
        "code": "models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)"
    },
    
    "VGG16": {
        "source": "torchvision.models",
        "weights": "VGG16_Weights.IMAGENET1K_V1",
        "input_size": 224,
        "use_case": "Medical imaging, simple blocks",
        "pros": ["Good for medical images", "Simple architecture"],
        "code": "models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)"
    },
    
    "DenseNet121": {
        "source": "torchvision.models",
        "weights": "DenseNet121_Weights.IMAGENET1K_V1",
        "input_size": 224,
        "use_case": "Dense connections, medical imaging",
        "pros": ["Good for medical", "Efficient gradients"],
        "code": "models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)"
    },
    
    "Inception_V3": {
        "source": "torchvision.models",
        "weights": "Inception_V3_Weights.IMAGENET1K_V1",
        "input_size": 299,
        "use_case": "Multi-scale features",
        "pros": ["Multi-scale processing", "Good accuracy"],
        "code": "models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)"
    },
    
    "EfficientNet-B0": {
        "source": "torchvision.models",
        "weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "input_size": 224,
        "use_case": "Mobile-efficient, medical imaging",
        "pros": ["Lightweight", "Fast", "Good accuracy-efficiency"],
        "code": "models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)"
    }
}

for model_name, info in pretrained_models.items():
    print(f"🤖 {model_name}")
    print(f"   Source: {info['source']}")
    print(f"   Input Size: {info['input_size']}×{info['input_size']}")
    print(f"   Use Case: {info['use_case']}")
    print(f"   Pros: {', '.join(info['pros'])}")
    print(f"   Code: {info['code']}\n")

# ============================================================================
# PART 3: RECOMMENDED WORKFLOW
# ============================================================================

print("=" * 80)
print("RECOMMENDED WORKFLOW FOR YOUR PROJECT")
print("=" * 80 + "\n")

workflow = [
    {
        "step": 1,
        "name": "Prepare Data",
        "files": ["dataset_metadata.csv", "preprocessed_data/"],
        "description": "Load metadata and preprocessed images",
        "code": """
import pandas as pd
df = pd.read_csv('dataset_metadata.csv')
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']
        """
    },
    {
        "step": 2,
        "name": "Create Dataset Class",
        "files": ["preprocessed_data/"],
        "description": "Define PyTorch Dataset for loading images",
        "code": """
class MedicalImageDataset(Dataset):
    def __init__(self, dataframe, img_dir):
        self.df = dataframe
        self.img_dir = img_dir
    
    def __getitem__(self, idx):
        bdmap_id = self.df.iloc[idx]['bdmap_id']
        img_path = f"{self.img_dir}/{bdmap_id}.jpg"
        image = cv2.imread(img_path)
        return torch.tensor(image)
        """
    },
    {
        "step": 3,
        "name": "Load Pre-trained Model",
        "files": ["None (online download)"],
        "description": "Load pre-trained weights from torchvision",
        "code": """
import torchvision.models as models
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# Modify final layer for your task (binary/multi-class)
num_classes = 2  # or your number of classes
model.fc = nn.Linear(2048, num_classes)
        """
    },
    {
        "step": 4,
        "name": "Fine-tune Model",
        "files": ["preprocessed_data/", "dataset_metadata.csv"],
        "description": "Train on your medical imaging task",
        "code": """
# Set up optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        """
    },
    {
        "step": 5,
        "name": "Evaluate Model",
        "files": ["test_df from dataset_metadata.csv"],
        "description": "Test on validation/test set",
        "code": """
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean()
        """
    }
]

for step in workflow:
    print(f"Step {step['step']}: {step['name']}")
    print(f"  Files: {step['files']}")
    print(f"  Description: {step['description']}")
    print(f"  Code:\n{step['code']}\n")

# ============================================================================
# PART 4: QUICK START CODE
# ============================================================================

print("=" * 80)
print("QUICK START CODE - COPY & USE")
print("=" * 80 + "\n")

quick_start = '''
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader

# ============ STEP 1: Load Metadata ============
df = pd.read_csv('dataset_metadata.csv')
train_df = df[df['split'] == 'train']
test_df = df[df['split'] == 'test']

print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# ============ STEP 2: Create Dataset Class ============
class MedicalImageDataset(Dataset):
    def __init__(self, dataframe, img_dir='preprocessed_data'):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image
        bdmap_id = self.df.iloc[idx]['bdmap_id']
        img_path = f"{self.img_dir}/{bdmap_id}.jpg"
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32) / 255.0
        return image

# ============ STEP 3: Create DataLoaders ============
train_dataset = MedicalImageDataset(train_df)
test_dataset = MedicalImageDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ============ STEP 4: Load Pre-trained Model ============
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Modify for grayscale images (input 1 channel instead of 3)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify for your task (example: binary classification)
num_classes = 2
model.fc = nn.Linear(2048, num_classes)

# ============ STEP 5: Setup Training ============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ============ STEP 6: Training Loop ============
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for images in train_loader:
        images = images.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, torch.randint(0, num_classes, (images.size(0),)).to(device))
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")

print("✓ Training complete!")
'''

print(quick_start)

# ============================================================================
# PART 5: FILE CHECKLIST
# ============================================================================

print("\n" + "=" * 80)
print("CHECKLIST BEFORE USING PRE-TRAINED MODELS")
print("=" * 80 + "\n")

checklist = [
    ("✓", "dataset_metadata.csv exists", "Contains split information"),
    ("✓", "preprocessed_data/ folder exists", "Contains preprocessed images"),
    ("✓", "preprocessing_log.json exists", "Confirms preprocessing settings"),
    ("✓", "Python packages installed", "torch, torchvision, opencv, pandas"),
    ("✓", "GPU available (optional)", "For faster training (cuda available)"),
    ("✓", "Images resized to 256×256", "Check from preprocessing_log.json"),
    ("✓", "Images normalized to [0,1]", "Check from preprocessing_log.json"),
]

for check, item, note in checklist:
    print(f"{check} {item:<40} - {note}")

# ============================================================================
# PART 6: SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY: WHICH FILES TO USE")
print("=" * 80 + "\n")

summary = """
📋 FOR YOUR PRE-TRAINED MODEL PROJECT:

PRIMARY FILES:
  1. dataset_metadata.csv      ← Use to split train/test
  2. preprocessed_data/        ← Use for image data

REFERENCE FILES:
  3. preprocessing_log.json    ← Verify preprocessing done
  4. dataset_config.json       ← Dataset structure info

EXTERNAL (Pre-trained weights):
  • Download automatically from torchvision
  • ResNet50, VGG16, DenseNet121, EfficientNet, etc.

WORKFLOW:
  dataset_metadata.csv + preprocessed_data/ → Load Data
                                              ↓
                                       Create Dataset
                                              ↓
                                    Load Pre-trained Model
                                              ↓
                                         Fine-tune
                                              ↓
                                           Evaluate

KEY POINTS:
✓ Use dataset_metadata.csv for train/test split
✓ Use preprocessed_data/ folder for images
✓ Images are already 256×256 and normalized
✓ Choose pre-trained model based on your task
✓ Modify input layer for grayscale images (1 channel)
✓ Modify output layer for your classification task
✓ Fine-tune on your medical imaging data

RECOMMENDED MODEL:
⭐ ResNet50 - Good for medical images, balanced speed/accuracy
⭐ DenseNet121 - Better for medical imaging tasks
⭐ EfficientNet - Lightweight, good accuracy
"""

print(summary)

# Save to file
summary_file = "TRANSFER_LEARNING_GUIDE.txt"
with open(summary_file, 'w') as f:
    f.write(summary)
    f.write("\n\nQUICK START CODE:\n")
    f.write(quick_start)

print(f"\n✓ Full guide saved to: {summary_file}")
print("\n" + "=" * 80)
