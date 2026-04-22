import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))
from scripts.Loading_Dataset import LiverDataset

from monai.networks.nets import SwinUNETR


# =========================================
# DEVICE
# =========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =========================================
# DATASET (ONLY TENSOR CONVERSION INSIDE DATASET)
# =========================================
root_dir = "../Dataset"

dataset = LiverDataset(root_dir=root_dir, transform=None)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


# =========================================
# MONAI SWIN UNETR MODEL
# =========================================
model = SwinUNETR(
    in_channels=3,
    out_channels=1,
    feature_size=48,
)

model = model.to(device)
model.eval()


# =========================================
# FEATURE EXTRACTION
# =========================================
all_features = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(dataloader, desc="Extracting features"):

        # images should already be tensors from dataset
        images = images.to(device)

        # forward through SwinViT encoder
        features = model.swinViT(images)

        # deepest feature map
        features = features[-1]   # [B, C, H, W]

        # global average pooling
        features = torch.mean(features, dim=[2, 3])  # [B, C]

        all_features.append(features.cpu())
        all_labels.append(labels)


# =========================================
# MERGE RESULTS
# =========================================
all_features = torch.cat(all_features, dim=0)
all_labels = torch.cat(all_labels, dim=0)


# =========================================
# OUTPUT
# =========================================
print("Feature shape:", all_features.shape)
print("Label shape:", all_labels.shape)

print("\nSample feature vector:")
print(all_features[0])