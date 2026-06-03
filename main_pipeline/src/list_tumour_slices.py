import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_pairs(data_dir):
    data_path = Path(data_dir)
    files = [
        path
        for path in data_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    mask_lookup = {path.stem: path for path in files if path.stem.endswith("_mask")}

    pairs = []
    for image_path in files:
        if image_path.stem.endswith("_mask"):
            continue
        mask_path = mask_lookup.get(f"{image_path.stem}_mask")
        if mask_path:
            pairs.append((image_path, mask_path))

    return sorted(pairs)


def mask_area_percent(mask_path):
    mask = Image.open(mask_path).convert("L")
    array = np.asarray(mask)
    return float((array > 0).mean() * 100)


def main(args):
    pairs = find_pairs(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    positive_count = 0
    negative_count = 0

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "mask_path", "has_tumour", "mask_area_percent"])

        for image_path, mask_path in pairs:
            area = mask_area_percent(mask_path)
            has_tumour = area > args.min_area_percent
            positive_count += int(has_tumour)
            negative_count += int(not has_tumour)
            writer.writerow([image_path, mask_path, has_tumour, f"{area:.4f}"])

    print(f"Checked {len(pairs)} image/mask pairs")
    print(f"Tumour-positive slices: {positive_count}")
    print(f"Tumour-negative slices: {negative_count}")
    print(f"Saved slice index to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/archive/kaggle_3m")
    parser.add_argument("--output", default="outputs/analysis/tumour_slice_index.csv")
    parser.add_argument("--min-area-percent", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
