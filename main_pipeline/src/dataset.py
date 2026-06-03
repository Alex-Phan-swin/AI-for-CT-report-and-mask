from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class BrainTumorSegmentationDataset(Dataset):
    def __init__(self, data_dir, image_size=256):
        self.data_dir = Path(data_dir)
        self.image_size = int(image_size)
        self.samples = self._find_samples()

        if not self.samples:
            raise RuntimeError(
                f"No image/mask pairs found in {self.data_dir}. "
                "Expected files like image.tif and image_mask.tif."
            )

    def _find_samples(self):
        files = [
            path
            for path in self.data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        mask_lookup = {path.stem: path for path in files if path.stem.endswith("_mask")}
        samples = []

        for image_path in files:
            if image_path.stem.endswith("_mask"):
                continue
            mask_path = mask_lookup.get(f"{image_path.stem}_mask")
            if mask_path:
                samples.append((image_path, mask_path))

        return sorted(samples)

    def __len__(self):
        return len(self.samples)

    def _load_grayscale(self, path):
        image = Image.open(path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    def _load_mask(self, path):
        mask = Image.open(path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        array = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        return {
            "image": self._load_grayscale(image_path),
            "mask": self._load_mask(mask_path),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }
