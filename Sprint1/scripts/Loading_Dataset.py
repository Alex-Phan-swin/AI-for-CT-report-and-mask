import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

class LiverDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # class folders (IMPORTANT: sorted for consistency)
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        # e.g. ['Unhealthy', 'healthy']

        self.image_paths = []
        self.labels = []

        #Assign numeric labels based on folder names, e.g. 'Unhealthy' -> 0, 'healthy' -> 1
        for label, cls in enumerate(self.classes):
            class_path = os.path.join(root_dir, cls)

            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    #Load image and label, apply transforms if any
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label