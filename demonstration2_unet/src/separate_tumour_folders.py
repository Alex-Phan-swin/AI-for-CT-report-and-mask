import argparse
import os
import shutil
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


def has_tumour(mask_path, min_area_percent):
    mask = Image.open(mask_path).convert("L")
    array = np.asarray(mask)
    area_percent = float((array > 0).mean() * 100)
    return area_percent > min_area_percent


def safe_link_or_copy(source, destination, use_copy):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    if use_copy:
        shutil.copy2(source, destination)
        return

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main(args):
    output_dir = Path(args.output_dir)
    pairs = find_pairs(args.data_dir)

    tumour_count = 0
    non_tumour_count = 0

    for image_path, mask_path in pairs:
        group = "tumour" if has_tumour(mask_path, args.min_area_percent) else "non_tumour"
        if group == "tumour":
            tumour_count += 1
        else:
            non_tumour_count += 1

        case_name = image_path.parent.name
        image_dest = output_dir / group / "images" / case_name / image_path.name
        mask_dest = output_dir / group / "masks" / case_name / mask_path.name

        safe_link_or_copy(image_path, image_dest, args.copy)
        safe_link_or_copy(mask_path, mask_dest, args.copy)

    print(f"Processed {len(pairs)} image/mask pairs")
    print(f"Tumour images: {tumour_count}")
    print(f"Non-tumour images: {non_tumour_count}")
    print(f"Saved organised folders under {output_dir}")
    if not args.copy:
        print("Used hard links where possible, so this should not duplicate dataset storage.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset/archive/kaggle_3m")
    parser.add_argument("--output-dir", default="dataset/sorted_by_tumour_status")
    parser.add_argument("--min-area-percent", type=float, default=0.0)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of using hard links. This uses more disk space.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
