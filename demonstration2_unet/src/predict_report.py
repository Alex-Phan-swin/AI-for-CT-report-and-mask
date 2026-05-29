import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model import UNet

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from qwen import BrainCTReportGenerator


def load_font(size, bold=False):
    """Cross-platform font loading that works on Windows, macOS, and Linux."""
    font_names = []
    if bold:
        font_names = ["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"]
    else:
        font_names = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf"]

    # Try to load each font in order
    for font_name in font_names:
        try:
            return ImageFont.truetype(font_name, size)
        except OSError:
            continue

    # Fallback to default font if none of the above work
    try:
        return ImageFont.load_default()
    except:
        # Last resort - create a basic font
        return ImageFont.load_default()


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(path, image_size):
    original = Image.open(path).convert("L")
    resized = original.resize((image_size, image_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    return original, resized, tensor


def save_mask(mask, output_path):
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
    mask_image.save(output_path)


def make_overlay(image, mask):
    base = image.convert("RGB")
    red = Image.new("RGB", base.size, (255, 0, 0))
    alpha = Image.fromarray((mask * 110).astype(np.uint8))
    overlay = Image.composite(red, base, alpha)

    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        draw = ImageDraw.Draw(overlay)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        draw.rectangle(bbox, outline=(255, 220, 0), width=3)

    return overlay


def mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return {
        "x_min": int(xs.min()),
        "y_min": int(ys.min()),
        "x_max": int(xs.max()),
        "y_max": int(ys.max()),
    }


def region_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return "no focal suspicious region"

    height, width = mask.shape
    cx = xs.mean()
    cy = ys.mean()

    horizontal = "left" if cx < width / 3 else "right" if cx > 2 * width / 3 else "central"
    vertical = "upper" if cy < height / 3 else "lower" if cy > 2 * height / 3 else "middle"

    if horizontal == "central" and vertical == "middle":
        return "central region"
    return f"{vertical}-{horizontal} region"


def build_evidence(mask, probability_map, area_threshold, probability_threshold, image_path, output_dir):
    area_percent = float(mask.mean() * 100)
    region = region_from_mask(mask)
    selected_probs = probability_map[mask > 0]
    mean_probability = float(selected_probs.mean()) if selected_probs.size else 0.0
    has_finding = area_percent >= area_threshold and mean_probability >= probability_threshold

    return {
        "input_image": str(image_path),
        "visual_evidence": {
            "predicted_mask": str(output_dir / "predicted_mask.png"),
            "overlay": str(output_dir / "overlay.png"),
            "source": "U-Net predicted segmentation mask",
        },
        "mask_derived_findings": {
            "finding_present": has_finding,
            "mask_area_percent": round(area_percent, 2),
            "primary_region": region,
            "bounding_box_pixels": mask_bbox(mask),
            "mean_mask_probability": round(mean_probability, 4),
            "area_threshold_percent": area_threshold,
            "mean_probability_threshold": probability_threshold,
        },
        "report_constraints": [
            "Report may only mention findings present in mask_derived_findings.",
            "Report must reference the predicted mask or overlay as visual evidence.",
            "Report must avoid unmeasured claims such as tumour type, grade, prognosis, or treatment.",
        ],
    }


def evidence_text(evidence):
    findings = evidence["mask_derived_findings"]
    bbox = findings["bounding_box_pixels"] or "None"
    return (
        "Structured Visual Evidence\n"
        "==========================\n\n"
        f"Input image: {evidence['input_image']}\n"
        f"Predicted mask: {evidence['visual_evidence']['predicted_mask']}\n"
        f"Overlay: {evidence['visual_evidence']['overlay']}\n"
        f"Evidence source: {evidence['visual_evidence']['source']}\n\n"
        f"Finding present: {'Yes' if findings['finding_present'] else 'No'}\n"
        f"Mask area: {findings['mask_area_percent']:.2f}%\n"
        f"Primary region: {findings['primary_region']}\n"
        f"Bounding box: {bbox}\n"
        f"Mean mask probability: {findings['mean_mask_probability']:.4f}\n\n"
        "Report constraints:\n"
        "- Only describe mask-derived findings.\n"
        "- Link the report to the visual overlay/mask.\n"
        "- Do not infer tumour type, grade, prognosis, or treatment.\n"
    )


def grounded_report(evidence):
    findings_data = evidence["mask_derived_findings"]
    area_percent = findings_data["mask_area_percent"]
    has_finding = findings_data["finding_present"]
    region = findings_data["primary_region"]
    mean_probability = findings_data["mean_mask_probability"]

    if has_finding:
        findings = (
            "A suspicious region was identified in the highlighted segmentation output. "
            f"The predicted mask occupies approximately {area_percent:.2f}% of the image area "
            f"and is localised primarily in the {region}. "
            f"The mean probability inside the predicted mask is {mean_probability:.2f}."
        )
        impression = (
            "The highlighted region is consistent with a possible tumour-like abnormality. "
            "This impression is grounded only in the predicted mask and overlay and requires clinical review."
        )
    else:
        findings = (
            "No suspicious region exceeded the configured visual-evidence thresholds. "
            f"The candidate abnormal mask area was {area_percent:.2f}%."
        )
        impression = (
            "No clear tumour-like abnormality was localised by the current model under the current thresholds. "
            "This does not rule out disease and requires clinical review."
        )

    return (
        "Grounded Segmentation Report\n"
        "============================\n\n"
        "Grounding method: report generated from structured segmentation evidence, not directly from the image alone.\n\n"
        f"Finding present: {'Yes' if has_finding else 'No'}\n"
        f"Mask area: {area_percent:.2f}%\n"
        f"Primary region: {region}\n\n"
        f"Findings:\n{findings}\n\n"
        f"Impression:\n{impression}\n\n"
        "Safety note: This prototype is for capstone demonstration only and is not a diagnostic tool.\n"
    )


def draw_wrapped_text(draw, position, text, font, fill, max_width, line_gap=6):
    x, y = position
    words = text.split()
    line = ""
    for word in words:
        trial = f"{line} {word}".strip()
        bbox = draw.textbbox((x, y), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            line = trial
        else:
            draw.text((x, y), line, font=font, fill=fill)
            y += font.size + line_gap
            line = word
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_gap
    return y


def draw_section(draw, x, y, title, text, title_font, body_font, width):
    draw.text((x, y), title, font=title_font, fill=(22, 28, 35))
    return draw_wrapped_text(
        draw,
        (x, y + title_font.size + 8),
        text,
        body_font,
        (35, 42, 50),
        width,
        line_gap=5,
    ) + 12


def make_demo_panel(original, mask_image, overlay, report, evidence, output_path):
    tile_size = (320, 320)
    margin = 28
    gap = 22
    report_width = 500
    title_height = 56
    panel_width = margin * 2 + tile_size[0] * 3 + gap * 3 + report_width
    panel_height = margin * 2 + title_height + tile_size[1] + 122

    panel = Image.new("RGB", (panel_width, panel_height), "white")
    draw = ImageDraw.Draw(panel)
    title_font = load_font(24, bold=True)
    label_font = load_font(17, bold=True)
    body_font = load_font(14)
    small_font = load_font(13)

    draw.text((margin, margin), "Visual-Grounded Medical Imaging Prototype", font=title_font, fill=(22, 28, 35))
    draw.text(
        (margin, margin + 34),
        "Image -> U-Net mask -> visual evidence -> constrained report",
        font=body_font,
        fill=(80, 87, 94),
    )

    y = margin + title_height
    x = margin
    items = [
        ("Input MRI", original.convert("RGB")),
        ("Predicted Mask", mask_image.convert("RGB")),
        ("Visual Evidence Overlay", overlay.convert("RGB")),
    ]

    for label, image in items:
        draw.text((x, y), label, font=label_font, fill=(22, 28, 35))
        panel.paste(image.resize(tile_size), (x, y + 28))
        x += tile_size[0] + gap

    report_x = x
    report_y = y
    draw.rectangle(
        (report_x, report_y, report_x + report_width, report_y + tile_size[1] + 92),
        outline=(210, 214, 220),
        width=1,
    )

    inner_x = report_x + 18
    inner_width = report_width - 36
    cursor_y = report_y + 16
    findings = evidence["mask_derived_findings"]

    draw.text((inner_x, cursor_y), "Grounded Report", font=label_font, fill=(22, 28, 35))
    cursor_y += 31
    draw_wrapped_text(
        draw,
        (inner_x, cursor_y),
        "Generated from structured segmentation evidence, not from free image-only report generation.",
        small_font,
        (80, 87, 94),
        inner_width,
        line_gap=4,
    )
    cursor_y += 46

    metric_gap = 12
    metric_width = (inner_width - metric_gap * 2) // 3
    metrics = [
        ("Finding", "Yes" if findings["finding_present"] else "No"),
        ("Mask area", f"{findings['mask_area_percent']:.2f}%"),
        ("Region", findings["primary_region"]),
    ]
    metric_x = inner_x
    for label, value in metrics:
        draw.rounded_rectangle(
            (metric_x, cursor_y, metric_x + metric_width, cursor_y + 52),
            radius=6,
            fill=(246, 248, 250),
            outline=(220, 224, 229),
            width=1,
        )
        draw.text((metric_x + 10, cursor_y + 7), label, font=small_font, fill=(90, 96, 104))
        draw_wrapped_text(
            draw,
            (metric_x + 10, cursor_y + 26),
            value,
            small_font,
            (22, 28, 35),
            metric_width - 20,
            line_gap=2,
        )
        metric_x += metric_width + metric_gap
    cursor_y += 70

    if findings["finding_present"]:
        findings_text = (
            "A suspicious region was identified in the highlighted segmentation output. "
            f"The predicted mask covers {findings['mask_area_percent']:.2f}% of the image and is localised "
            f"primarily in the {findings['primary_region']}."
        )
        impression_text = (
            "The highlighted region is consistent with a possible tumour-like abnormality. "
            "This statement is grounded only in the mask and overlay."
        )
    else:
        findings_text = (
            "No suspicious region exceeded the configured visual-evidence thresholds. "
            f"The candidate abnormal mask area was {findings['mask_area_percent']:.2f}%."
        )
        impression_text = (
            "No clear tumour-like abnormality was localised by the current model under the current thresholds. "
            "Clinical review is still required."
        )

    cursor_y = draw_section(
        draw,
        inner_x,
        cursor_y,
        "Findings",
        findings_text,
        label_font,
        body_font,
        inner_width,
    )
    cursor_y = draw_section(
        draw,
        inner_x,
        cursor_y,
        "Impression",
        impression_text,
        label_font,
        body_font,
        inner_width,
    )
    draw_section(
        draw,
        inner_x,
        cursor_y,
        "Safety",
        "Prototype only. Not a diagnostic tool. Requires clinical review.",
        label_font,
        body_font,
        inner_width,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


# def main(args):
#     # Load checkpoint with weights_only=True for security, then get metadata separately
#     checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
#     # Load metadata separately (not secure but needed for image_size)
#     checkpoint_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
#     image_size = int(checkpoint_meta.get("image_size", args.image_size))

#     device = choose_device()
#     model = UNet().to(device)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()

#     original, resized, tensor = load_image(args.image, image_size)
#     tensor = tensor.to(device)

#     with torch.no_grad():
#         logits = model(tensor)
#         prob = torch.sigmoid(logits).squeeze().cpu().numpy()

#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)

#     mask = (prob >= args.pixel_probability_threshold).astype(np.uint8)
#     evidence = build_evidence(
#         mask,
#         prob,
#         args.area_threshold,
#         args.mean_probability_threshold,
#         args.image,
#         output_dir,
#     )
#     display_mask = mask if evidence["mask_derived_findings"]["finding_present"] else np.zeros_like(mask)

#     original_output = output_dir / "original.png"
#     mask_output = output_dir / "predicted_mask.png"
#     overlay_output = output_dir / "overlay.png"
#     evidence_json_output = output_dir / "evidence.json"
#     evidence_text_output = output_dir / "evidence.txt"
#     report_output = output_dir / "report.txt"
#     panel_output = output_dir / "demo_panel.png"

#     original.save(original_output)
#     save_mask(display_mask, output_dir / "predicted_mask.png")
#     overlay = make_overlay(resized, display_mask)
#     overlay.save(overlay_output)

#     evidence_json_output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
#     evidence_text_output.write_text(evidence_text(evidence), encoding="utf-8")

#     report = grounded_report(evidence)
#     report_output.write_text(report, encoding="utf-8")
#     make_demo_panel(resized, Image.fromarray(display_mask * 255), overlay, report, evidence, panel_output)

#     print(f"Saved original, mask, overlay, evidence, report, and demo panel to {output_dir}")
#     print(report)


def main(args):
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    image_size = int(checkpoint_meta.get("image_size", args.image_size))

    device = choose_device()
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load image
    original, resized, tensor = load_image(args.image, image_size)
    tensor = tensor.to(device)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mask
    mask = (prob >= args.pixel_probability_threshold).astype(np.uint8)

    # Evidence
    evidence = build_evidence(
        mask,
        prob,
        args.area_threshold,
        args.mean_probability_threshold,
        args.image,
        output_dir,
    )

    display_mask = mask if evidence["mask_derived_findings"]["finding_present"] else np.zeros_like(mask)

    # Save artifacts
    original_output = output_dir / "original.png"
    mask_output = output_dir / "predicted_mask.png"
    overlay_output = output_dir / "overlay.png"
    evidence_json_output = output_dir / "evidence.json"
    evidence_text_output = output_dir / "evidence.txt"
    report_output = output_dir / "report.txt"
    panel_output = output_dir / "demo_panel.png"

    original.save(original_output)
    save_mask(display_mask, mask_output)
    overlay = make_overlay(resized, display_mask)
    overlay.save(overlay_output)

    evidence_json_output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    evidence_text_output.write_text(evidence_text(evidence), encoding="utf-8")

    # ================================
    # 🔥 NEW: Qwen REPORT GENERATION
    # ================================

    report_generator = BrainCTReportGenerator()

    report = report_generator.generate_report(
        image_path=args.image,
        evidence=evidence
    )

    report_output.write_text(report, encoding="utf-8")

    # Demo panel (still uses same report text)
    make_demo_panel(
        resized,
        Image.fromarray(display_mask * 255),
        overlay,
        report,
        evidence,
        panel_output
    )

    print(f"Saved outputs to {output_dir}")
    print("\n=== GENERATED REPORT ===\n")
    print(report)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pixel-probability-threshold", type=float, default=0.5)
    parser.add_argument("--area-threshold", type=float, default=1.0)
    parser.add_argument("--mean-probability-threshold", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
