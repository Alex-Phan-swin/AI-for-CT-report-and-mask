import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def find_demo_image(input_dir):
    input_path = Path(input_dir)
    images = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and not path.stem.endswith("_mask")
    )

    if not images:
        raise RuntimeError(
            f"No demo image found in {input_path}. "
            "Add one .tif, .png, .jpg, .jpeg, or .bmp image."
        )

    if len(images) > 1:
        names = "\n".join(f"- {path.name}" for path in images)
        raise RuntimeError(
            "More than one demo image found. Keep exactly one image in the demo folder:\n"
            f"{names}"
        )

    return images[0]


def find_matching_mask(image_path, mask_roots):
    expected_name = f"{image_path.stem}_mask{image_path.suffix}"
    for root in mask_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        matches = sorted(root_path.rglob(expected_name))
        if matches:
            return matches[0]
    return None


def make_ground_truth_overlay(image, mask):
    base = image.convert("RGB")
    mask_array = (np.asarray(mask.convert("L")) > 0).astype(np.uint8)
    red = Image.new("RGB", base.size, (255, 0, 0))
    alpha = Image.fromarray((mask_array * 110).astype(np.uint8))
    overlay = Image.composite(red, base, alpha)

    ys, xs = np.where(mask_array > 0)
    if len(xs) > 0 and len(ys) > 0:
        draw = ImageDraw.Draw(overlay)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        draw.rectangle(bbox, outline=(255, 220, 0), width=3)

    return overlay


def make_ground_truth_panel(image_path, mask_path, output_path):
    image = Image.open(image_path).convert("L").resize((320, 320), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((320, 320), Image.NEAREST)
    overlay = make_ground_truth_overlay(image, mask)

    tile_size = (320, 320)
    margin = 28
    gap = 24
    title_height = 54
    width = margin * 2 + tile_size[0] * 3 + gap * 2
    height = margin * 2 + title_height + tile_size[1] + 34

    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)

    # #Fonts for mac
    # # title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
    # # label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17)

    # #Fonts for windows
    # try:
    #     title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
    #     label_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 17)
    # except Exception as e:
    #     print(f'Error loading fonts: {e}, if on mac switch to mac font')


        # Try Windows fonts first, then fall back to Mac fonts
    try:
        title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
        label_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 17)
        body_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
        small_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)

    except OSError:
        print("Windows fonts not found. Trying Mac fonts...")

        title_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24
        )
        label_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17
        )
        body_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 14
        )
        small_font = ImageFont.truetype(
            "/System/Library/Fonts/Supplemental/Arial.ttf", 13
        )


    draw.text((margin, margin), "Dataset Label Preview", font=title_font, fill=(22, 28, 35))
    draw.text(
        (margin, margin + 32),
        "MRI image with ground-truth segmentation mask from the labelled dataset",
        font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15),
        fill=(80, 87, 94),
    )

    x = margin
    y = margin + title_height
    items = [
        ("MRI Image", image.convert("RGB")),
        ("Ground-Truth Mask", mask.convert("RGB")),
        ("Ground-Truth Overlay", overlay),
    ]

    for label, tile in items:
        draw.text((x, y), label, font=label_font, fill=(22, 28, 35))
        panel.paste(tile, (x, y + 28))
        x += tile_size[0] + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def main(args):
    image_path = find_demo_image(args.input_dir)
    print(f"Analysing demo image: {image_path}")

    command = [
        sys.executable,
        "src/predict_report.py",
        "--checkpoint",
        args.checkpoint,
        "--image",
        str(image_path),
        "--output-dir",
        args.output_dir,
    ]

    subprocess.run(command, check=True)

    mask_path = find_matching_mask(image_path, args.mask_roots)
    if mask_path:
        ground_truth_output = Path(args.output_dir) / "ground_truth_panel.png"
        make_ground_truth_panel(image_path, mask_path, ground_truth_output)
        print(f"Saved ground-truth comparison panel to {ground_truth_output}")
    else:
        print("No matching ground-truth mask found. Skipped ground_truth_panel.png.")

    print("\nOne-command demo outputs:")
    for name in [
        "original.png",
        "predicted_mask.png",
        "overlay.png",
        "probability_heatmap.png",
        "evidence.json",
        "evidence.txt",
        "report.txt",
        "report_validation.json",
        "report_validation.txt",
        "demo_panel.png",
        "sprint3_demo.html",
        "ground_truth_panel.png",
    ]:
        path = Path(args.output_dir) / name
        if path.exists():
            print(f"- {path}")

    if args.open:
        panel_path = Path(args.output_dir) / args.open_file
        if not panel_path.exists():
            raise RuntimeError(f"Cannot open missing output file: {panel_path}")
        if sys.platform == "darwin":
            subprocess.run(["open", str(panel_path)], check=True)
        elif os.name == "nt":
            os.startfile(panel_path)
        else:
            subprocess.run(["xdg-open", str(panel_path)], check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="demo_input")
    parser.add_argument("--checkpoint", default="models/unet_brain_mri.pth")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--mask-roots",
        nargs="*",
        default=[
            "dataset/sorted_by_tumour_status/tumour/masks",
            "dataset/sorted_by_tumour_status/non_tumour/masks",
            "dataset/archive/kaggle_3m",
        ],
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated demo image after analysis.",
    )
    parser.add_argument(
        "--open-file",
        default="demo_panel.png",
        help="Output file to open when --open is used.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
