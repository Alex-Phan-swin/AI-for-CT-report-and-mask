import argparse
import os
import subprocess
import sys
import random
import platform
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# =========================
# FONT LOADER
# =========================
def load_font(size, bold=False):
    font_names = (
        ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]
    )

    for name in font_names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue

    return ImageFont.load_default()


# =========================
# DEMO IMAGE FINDER
# =========================
def find_demo_image(input_dir):
    input_path = Path(input_dir)

    images = [
        p for p in input_path.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and "_mask" not in p.stem
    ]

    if not images:
        raise RuntimeError(f"No demo images in {input_dir}")

    return images[0]


# =========================
# MASK MATCHING
# =========================
def find_matching_mask(image_path, mask_roots):
    expected_name = f"{image_path.stem}_mask{image_path.suffix}"

    for root in mask_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue

        match = next(root_path.rglob(expected_name), None)
        if match:
            return match

    return None


# =========================
# OVERLAY
# =========================
def make_ground_truth_overlay(image, mask):
    base = image.convert("RGB")

    mask_array = (np.asarray(mask.convert("L")) > 0).astype(np.uint8)

    red_layer = Image.new("RGB", base.size, (255, 0, 0))
    alpha = Image.fromarray((mask_array * 120).astype(np.uint8))

    overlay = Image.composite(red_layer, base, alpha)

    ys, xs = np.where(mask_array > 0)
    if len(xs) > 0:
        draw = ImageDraw.Draw(overlay)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        draw.rectangle(bbox, outline=(255, 215, 0), width=3)

    return overlay


# =========================
# PANEL
# =========================
def make_ground_truth_panel(image_path, mask_path, output_path):
    image = Image.open(image_path).convert("L").resize((320, 320))
    mask = Image.open(mask_path).convert("L").resize((320, 320))
    overlay = make_ground_truth_overlay(image, mask)

    panel = Image.new("RGB", (1100, 450), "white")
    draw = ImageDraw.Draw(panel)

    title_font = load_font(22, bold=True)
    label_font = load_font(16, bold=True)

    draw.text((25, 20), "Ground Truth Visual Evidence", font=title_font, fill=(20, 20, 20))

    items = [
        ("Input Scan", image.convert("RGB")),
        ("Segmentation Mask", mask.convert("RGB")),
        ("Overlay", overlay),
    ]

    x = 25
    y = 80

    for label, img in items:
        draw.text((x, y - 25), label, font=label_font, fill=(40, 40, 40))
        panel.paste(img, (x, y))
        x += 350

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


# =========================
# AUTO OPEN FUNCTION
# =========================
def open_path(path: Path):
    if not path.exists():
        return

    system = platform.system()

    if system == "Windows":
        os.startfile(str(path))
    elif system == "Darwin":
        subprocess.run(["open", str(path)])
    else:
        subprocess.run(["xdg-open", str(path)])


# =========================
# MAIN PIPELINE STEP
# =========================
def main(args):

    demo_image = find_demo_image(args.input_dir)
    print(f"Analysing demo image: {demo_image}")

    command = [
        sys.executable,
        "src/predict_report.py",
        "--checkpoint",
        args.checkpoint,
        "--image",
        str(demo_image),
        "--output-dir",
        args.output_dir,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)

    subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).resolve().parent.parent,
        env=env
    )

    output_dir = Path(args.output_dir)

    # =========================
    # Ground truth panel
    # =========================
    mask = find_matching_mask(demo_image, args.mask_roots)

    panel_path = None
    if mask:
        panel_path = output_dir / "ground_truth_panel.png"
        make_ground_truth_panel(demo_image, mask, panel_path)
        print(f"Saved: {panel_path}")

    # =========================
    # REPORT PATH
    # =========================
    report_path = output_dir / "report.txt"

    print("\nOutputs:")
    for f in output_dir.glob("*"):
        print("-", f)

    # =========================
    # AUTO OPEN EVERYTHING
    # =========================
    print("\nOpening outputs...")

    open_path(report_path)

    if panel_path:
        open_path(panel_path)

    open_path(output_dir)


# =========================
# ARGS
# =========================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input-dir", default="demo_input")
    p.add_argument("--checkpoint", default="models/unet_brain_mri.pth")
    p.add_argument("--output-dir", default="outputs/demo")

    p.add_argument("--mask-roots", nargs="*", default=[
        "dataset/sorted_by_tumour_status/tumour/masks",
        "dataset/sorted_by_tumour_status/non_tumour/masks",
        "dataset/archive/kaggle_3m",
    ])
    p.add_argument(
        "--open",
        action="store_true",
        help="Pipeline compatibility flag (ignored, auto-opens outputs anyway)"
)

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())