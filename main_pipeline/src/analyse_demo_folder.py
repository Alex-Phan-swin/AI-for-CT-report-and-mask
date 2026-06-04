import argparse
import os
import subprocess
import sys
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
# GROUND TRUTH OVERLAY (Simple, just for comparison)
# =========================
def make_ground_truth_overlay(image, mask):
    """Create ground truth overlay with matching sizes"""
    # Ensure both images are the same size
    base = image.convert("RGB")
    
    # Resize mask to match base image size
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.NEAREST)
    
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
# SIMPLIFIED PANEL (Only shows colour-coded overlay + legend)
# =========================
def make_demo_panel(image_path, colour_overlay_path, legend_path, output_path):
    """Create a clean demo panel with colour-coded overlay and legend"""
    
    original = Image.open(image_path).convert("RGB").resize((320, 320))
    colour_overlay = Image.open(colour_overlay_path).resize((320, 320))
    legend = Image.open(legend_path)
    
    # Create panel
    panel = Image.new("RGB", (900, 450), "white")
    draw = ImageDraw.Draw(panel)
    
    title_font = load_font(22, bold=True)
    label_font = load_font(16, bold=True)
    
    draw.text((25, 20), "Brain Tumour Segmentation - Colour-Coded Evidence", font=title_font, fill=(20, 20, 20))
    
    # Layout
    items = [
        ("Original Scan", original),
        ("Colour-Coded Segmentation", colour_overlay),
    ]
    
    x = 25
    y = 80
    
    for label, img in items:
        draw.text((x, y - 25), label, font=label_font, fill=(40, 40, 40))
        panel.paste(img, (x, y))
        x += 350
    
    # Paste legend in bottom right
    legend_resized = legend.resize((250, 150))
    panel.paste(legend_resized, (620, 280))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    return output_path


# =========================
# AUTO OPEN FUNCTION
# =========================
def open_path(path: Path):
    if not path.exists():
        print(f"Warning: {path} does not exist")
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

    # Find demo image
    demo_image = find_demo_image(args.input_dir)
    print(f"Analysing demo image: {demo_image}")

    # Run prediction with colour-coded output (using new predict_report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    command = [
        sys.executable,
        "src/predict_report.py",
        "--checkpoint", args.checkpoint,
        "--image", str(demo_image),
        "--output-dir", str(output_dir),
        "--threshold", str(args.threshold),
    ]
    
    # Add Qwen flag if requested
    if args.use_qwen:
        command.append("--use-qwen")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)

    print(f"\nRunning prediction...")
    result = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent, env=env)
    
    if result.returncode != 0:
        print("Prediction failed!")
        return

    # Find the generated outputs
    colour_overlay = output_dir / "colour_coded_overlay.png"
    legend = output_dir / "colour_legend.png"
    grounded_report = output_dir / "grounded_report.txt"
    
    # Check if colour-coded outputs exist
    if not colour_overlay.exists():
        print(f"Warning: colour_coded_overlay.png not found in {output_dir}")
        print("Falling back to basic overlay...")
        colour_overlay = output_dir / "basic_overlay.png"
    
    # =========================
    # Ground truth panel (if mask available)
    # =========================
    mask = find_matching_mask(demo_image, args.mask_roots)
    gt_panel_path = None
    
    if mask:
        gt_panel_path = output_dir / "ground_truth_comparison.png"
        original_img = Image.open(demo_image).convert("L").resize((320, 320))
        gt_overlay = make_ground_truth_overlay(original_img, Image.open(mask))
        
        # Simple comparison panel
        comparison = Image.new("RGB", (700, 350), "white")
        draw = ImageDraw.Draw(comparison)
        draw.text((25, 20), "Ground Truth Comparison", font=load_font(20, bold=True), fill=(20, 20, 20))
        comparison.paste(original_img.convert("RGB"), (25, 80))
        comparison.paste(gt_overlay, (375, 80))
        draw.text((25, 55), "Original", font=load_font(12), fill=(40, 40, 40))
        draw.text((375, 55), "Ground Truth Mask", font=load_font(12), fill=(40, 40, 40))
        comparison.save(gt_panel_path)
        print(f"Saved ground truth comparison: {gt_panel_path}")
    
    # =========================
    # Create demo panel with colour-coded output
    # =========================
    demo_panel_path = output_dir / "demo_panel.png"
    make_demo_panel(demo_image, colour_overlay, legend, demo_panel_path)
    print(f"Saved demo panel: {demo_panel_path}")
    
    # =========================
    # OPEN ONLY THE ESSENTIAL FILES
    # =========================
    print("\n" + "=" * 50)
    print("OPENING DELIVERABLES")
    print("=" * 50)
    
    # 1. Open the colour-coded overlay (main deliverable)
    if colour_overlay.exists():
        print(f"Opening colour-coded overlay...")
        open_path(colour_overlay)
    
    # 2. Open the grounded report (main deliverable)
    if grounded_report.exists():
        print(f"Opening grounded report...")
        open_path(grounded_report)
    
    # 3. Optionally open the demo panel
    if args.show_panel:
        print(f"Opening demo panel...")
        open_path(demo_panel_path)
    
    # 4. Open output directory for reference
    if args.open_folder:
        print(f"Opening output folder...")
        open_path(output_dir)
    
    # Print summary
    print("\n" + "=" * 50)
    print("OUTPUT SUMMARY")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print("\nKey deliverables:")
    print(f"  • Colour-coded overlay: {colour_overlay.name}")
    print(f"  • Grounded report: {grounded_report.name}")
    print(f"  • Colour legend: {legend.name}")
    
    if gt_panel_path:
        print(f"  • Ground truth comparison: {gt_panel_path.name}")
    
    print("\n✓ Demo complete. The report references specific colours shown in the overlay.")


# =========================
# ARGS
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Run colour-coded segmentation demo")
    
    p.add_argument("--input-dir", default="demo_input", help="Directory with demo images")
    p.add_argument("--checkpoint", default="models/unet_brain_mri.pth", help="Model checkpoint")
    p.add_argument("--output-dir", default="outputs/demo", help="Output directory")
    p.add_argument("--threshold", type=float, default=0.15, help="Segmentation threshold")
    
    p.add_argument("--mask-roots", nargs="*", default=[
        "dataset/sorted_by_tumour_status/tumour/masks",
        "dataset/sorted_by_tumour_status/non_tumour/masks",
        "dataset/archive/kaggle_3m",
    ], help="Directories to search for ground truth masks")
    
    # Qwen option
    p.add_argument("--use-qwen", action="store_true", help="Use Qwen for report formatting")
    
    # Display options
    p.add_argument("--show-panel", action="store_true", help="Also open the demo panel")
    p.add_argument("--open-folder", action="store_true", help="Open output folder")
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("BRAIN TUMOUR SEGMENTATION DEMO - COLOUR-CODED EVIDENCE")
    print("=" * 60)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Qwen: {'Enabled' if args.use_qwen else 'Disabled'}")
    print("=" * 60 + "\n")
    
    main(args)