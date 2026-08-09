import argparse
import html
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from model import UNet
from report_generator import write_report_outputs


REGION_PALETTE = [
    {"name": "red", "hex": "#ff3b30", "rgb": (255, 59, 48)},
    {"name": "blue", "hex": "#0a84ff", "rgb": (10, 132, 255)},
    {"name": "green", "hex": "#30d158", "rgb": (48, 209, 88)},
    {"name": "orange", "hex": "#ff9f0a", "rgb": (255, 159, 10)},
    {"name": "purple", "hex": "#bf5af2", "rgb": (191, 90, 242)},
]


def assign_region_colors(regions):
    for index, region in enumerate(regions, start=1):
        region["id"] = f"R{index}"
        region["color"] = REGION_PALETTE[(index - 1) % len(REGION_PALETTE)]
    return regions


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


def make_region_mask(mask_shape, box):
    region_mask = np.zeros(mask_shape, dtype=np.uint8)
    region_mask[
        box["y_min"] : box["y_max"] + 1,
        box["x_min"] : box["x_max"] + 1,
    ] = 1
    return region_mask


def make_overlay(image, mask, regions=None):
    base = image.convert("RGB")
    overlay = base
    regions = regions or []

    if regions:
        for region in regions:
            box = region["bounding_box_pixels"]
            region_mask = make_region_mask(mask.shape, box) * mask
            color = tuple(region["color"]["rgb"])
            color_layer = Image.new("RGB", base.size, color)
            alpha = Image.fromarray((region_mask * 125).astype(np.uint8))
            overlay = Image.composite(color_layer, overlay, alpha)
    else:
        red = Image.new("RGB", base.size, REGION_PALETTE[0]["rgb"])
        alpha = Image.fromarray((mask * 110).astype(np.uint8))
        overlay = Image.composite(red, overlay, alpha)

    ys, xs = np.where(mask > 0)
    if len(xs) > 0 and len(ys) > 0:
        draw = ImageDraw.Draw(overlay)
        if regions:
            try:
                label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 14)
            except OSError:
                label_font = ImageFont.load_default()
            for region in regions:
                box = region["bounding_box_pixels"]
                bbox = (box["x_min"], box["y_min"], box["x_max"], box["y_max"])
                color = tuple(region["color"]["rgb"])
                draw.rectangle(bbox, outline=color, width=3)
                label = f"{region['id']} {region['color']['name']} {region['mean_probability']:.2f}"
                raw_label_box = draw.textbbox((0, 0), label, font=label_font)
                label_width = raw_label_box[2] - raw_label_box[0]
                label_height = raw_label_box[3] - raw_label_box[1]
                label_x = min(max(0, bbox[0]), max(0, overlay.size[0] - label_width - 2))
                label_y = max(0, bbox[1] - label_height - 4)
                label_box = draw.textbbox((label_x, label_y), label, font=label_font)
                draw.rectangle(label_box, fill=color)
                draw.text((label_x, label_y), label, font=label_font, fill=(255, 255, 255))
        else:
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


def connected_regions(mask, probability_map, min_pixels=12, max_regions=5):
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    regions = []

    for y in range(height):
        for x in range(width):
            if mask[y, x] == 0 or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            pixels = []

            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if len(pixels) < min_pixels:
                continue

            ys = np.array([pixel[0] for pixel in pixels])
            xs = np.array([pixel[1] for pixel in pixels])
            region_mask = np.zeros(mask.shape, dtype=np.uint8)
            region_mask[ys, xs] = 1
            probs = probability_map[ys, xs]
            area_percent = float(len(pixels) / mask.size * 100)
            regions.append(
                {
                    "pixel_count": int(len(pixels)),
                    "area_percent": round(area_percent, 2),
                    "mean_probability": round(float(probs.mean()), 4),
                    "max_probability": round(float(probs.max()), 4),
                    "region_label": region_from_mask(region_mask),
                    "centroid_pixels": {
                        "x": round(float(xs.mean()), 1),
                        "y": round(float(ys.mean()), 1),
                    },
                    "bounding_box_pixels": {
                        "x_min": int(xs.min()),
                        "y_min": int(ys.min()),
                        "x_max": int(xs.max()),
                        "y_max": int(ys.max()),
                    },
                }
            )

    regions.sort(key=lambda item: (item["area_percent"], item["mean_probability"]), reverse=True)
    return assign_region_colors(regions[:max_regions])


def make_region_record(region_mask, probability_map, description):
    ys, xs = np.where(region_mask > 0)
    if len(xs) == 0:
        return None

    probs = probability_map[ys, xs]
    area_percent = float(len(xs) / region_mask.size * 100)
    return {
        "pixel_count": int(len(xs)),
        "area_percent": round(area_percent, 2),
        "mean_probability": round(float(probs.mean()), 4),
        "max_probability": round(float(probs.max()), 4),
        "region_label": f"{description} of {region_from_mask(region_mask)}",
        "centroid_pixels": {
            "x": round(float(xs.mean()), 1),
            "y": round(float(ys.mean()), 1),
        },
        "bounding_box_pixels": {
            "x_min": int(xs.min()),
            "y_min": int(ys.min()),
            "x_max": int(xs.max()),
            "y_max": int(ys.max()),
        },
        "region_type": "internal_subregion",
    }


def split_single_region(mask, probability_map, max_subregions=3, min_pixels=12):
    bbox = mask_bbox(mask)
    if bbox is None or int(mask.sum()) < min_pixels * 2:
        return []

    width = bbox["x_max"] - bbox["x_min"] + 1
    height = bbox["y_max"] - bbox["y_min"] + 1
    split_axis = "x" if width >= height else "y"
    start = bbox[f"{split_axis}_min"]
    stop = bbox[f"{split_axis}_max"] + 1
    split_count = min(max_subregions, max(2, stop - start))
    edges = np.linspace(start, stop, split_count + 1, dtype=int)
    descriptions = ["left-side", "central", "right-side"] if split_axis == "x" else ["upper", "middle", "lower"]

    subregions = []
    for index in range(split_count):
        lo = int(edges[index])
        hi = int(edges[index + 1])
        if hi <= lo:
            continue

        region_mask = np.zeros(mask.shape, dtype=np.uint8)
        if split_axis == "x":
            region_mask[:, lo:hi] = mask[:, lo:hi]
        else:
            region_mask[lo:hi, :] = mask[lo:hi, :]

        if int(region_mask.sum()) < min_pixels:
            continue

        description = descriptions[index] if index < len(descriptions) else f"subregion {index + 1}"
        record = make_region_record(region_mask, probability_map, description)
        if record:
            subregions.append(record)

    if len(subregions) < 2:
        return []

    return assign_region_colors(subregions)


def evidence_regions(mask, probability_map):
    regions = connected_regions(mask, probability_map)
    if len(regions) == 1:
        split_regions = split_single_region(mask, probability_map)
        if split_regions:
            return split_regions
    return regions


def make_probability_heatmap(image, probability_map, regions):
    base = image.convert("RGB")
    prob = np.clip(probability_map, 0.0, 1.0)
    red = Image.new("RGB", base.size, REGION_PALETTE[0]["rgb"])
    alpha = Image.fromarray((prob * 150).astype(np.uint8))
    heatmap = Image.composite(red, base, alpha)
    return make_overlay(heatmap, (prob >= 0.5).astype(np.uint8), regions)


def build_evidence(mask, probability_map, area_threshold, probability_threshold, image_path, output_dir):
    area_percent = float(mask.mean() * 100)
    region = region_from_mask(mask)
    selected_probs = probability_map[mask > 0]
    mean_probability = float(selected_probs.mean()) if selected_probs.size else 0.0
    has_finding = area_percent >= area_threshold and mean_probability >= probability_threshold
    regions = evidence_regions(mask, probability_map)

    return {
        "input_image": str(image_path),
        "visual_evidence": {
            "predicted_mask": str(output_dir / "predicted_mask.png"),
            "overlay": str(output_dir / "overlay.png"),
            "probability_heatmap": str(output_dir / "probability_heatmap.png"),
            "source": "U-Net predicted segmentation mask",
            "colour_key": {
                region["id"]: {
                    "colour": region["color"]["name"],
                    "hex": region["color"]["hex"],
                    "meaning": f"{region['id']} evidence region in the predicted segmentation mask",
                }
                for region in regions
            },
        },
        "evidence_regions": regions if has_finding else [],
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
    regions = evidence.get("evidence_regions", [])
    region_lines = "\n".join(
        "- {id}: {area_percent:.2f}% area, mean probability {mean_probability:.4f}, "
        "{region_label}, bbox {bounding_box_pixels}".format(**region)
        for region in regions
    ) or "- None above evidence thresholds"
    return (
        "Structured Visual Evidence\n"
        "==========================\n\n"
        f"Input image: {evidence['input_image']}\n"
        f"Predicted mask: {evidence['visual_evidence']['predicted_mask']}\n"
        f"Overlay: {evidence['visual_evidence']['overlay']}\n"
        f"Probability heatmap: {evidence['visual_evidence']['probability_heatmap']}\n"
        f"Evidence source: {evidence['visual_evidence']['source']}\n\n"
        f"Finding present: {'Yes' if findings['finding_present'] else 'No'}\n"
        f"Mask area: {findings['mask_area_percent']:.2f}%\n"
        f"Primary region: {findings['primary_region']}\n"
        f"Bounding box: {bbox}\n"
        f"Mean mask probability: {findings['mean_mask_probability']:.4f}\n\n"
        "Evidence regions:\n"
        f"{region_lines}\n\n"
        "Report constraints:\n"
        "- Only describe mask-derived findings.\n"
        "- Link the report to the visual overlay/mask.\n"
        "- Do not infer tumour type, grade, prognosis, or treatment.\n"
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

    # #mac fonts
    # # title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 24)
    # # label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 17)
    # # body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    # # small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
    
    # #window
    # title_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 24)
    # label_font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 17)
    # body_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    # small_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)

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
    regions = evidence.get("evidence_regions", [])

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
        ("Regions", str(len(regions))),
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

    if regions:
        draw.text((inner_x, cursor_y), "Evidence Regions", font=label_font, fill=(22, 28, 35))
        cursor_y += 28
        for region in regions[:3]:
            line = (
                f"{region['id']} | {region['area_percent']:.2f}% area | "
                f"{region['mean_probability']:.2f} mean prob | {region['region_label']}"
            )
            cursor_y = draw_wrapped_text(
                draw,
                (inner_x, cursor_y),
                line,
                small_font,
                (35, 42, 50),
                inner_width,
                line_gap=3,
            )
        cursor_y += 8

    if findings["finding_present"]:
        evidence_id = regions[0]["id"] if regions else "mask"
        findings_text = (
            f"A suspicious region was identified in evidence region {evidence_id}. "
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


def write_html_demo(evidence, report, validation, output_dir):
    output_path = Path(output_dir)
    findings = evidence["mask_derived_findings"]
    regions = evidence.get("evidence_regions", [])
    region_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(region['id'])}</td>"
        f"<td>{region['area_percent']:.2f}%</td>"
        f"<td>{region['mean_probability']:.2f}</td>"
        f"<td>{html.escape(region['region_label'])}</td>"
        f"<td>{html.escape(str(region['bounding_box_pixels']))}</td>"
        "</tr>"
        for region in regions
    ) or '<tr><td colspan="5">No evidence region exceeded the configured thresholds.</td></tr>'
    claim_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['status'].upper())}</td>"
        f"<td>{html.escape(row['sentence'])}</td>"
        f"<td>{html.escape(row['evidence_id'])}</td>"
        f"<td>{html.escape(row['support'])}</td>"
        "</tr>"
        for row in validation.get("sentence_evidence_map", [])
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sprint 3 Grounded Report Demo</title>
  <style>
    :root {{ color-scheme: light; font-family: Arial, Helvetica, sans-serif; }}
    body {{ margin: 0; background: #f6f8fa; color: #1f2933; }}
    header {{ padding: 24px 32px 14px; background: #ffffff; border-bottom: 1px solid #d9dee5; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    main {{ padding: 24px 32px 36px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 16px; }}
    .panel {{ background: #ffffff; border: 1px solid #d9dee5; border-radius: 8px; padding: 16px; }}
    .image-panel img {{ width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: #111827; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ text-align: left; border-top: 1px solid #e3e7ed; padding: 9px 8px; vertical-align: top; }}
    th {{ background: #f1f4f7; }}
    pre {{ white-space: pre-wrap; margin: 0; font: 14px/1.45 Menlo, Consolas, monospace; }}
    .stack {{ display: grid; gap: 16px; margin-top: 16px; grid-template-columns: 1fr 1fr; }}
    @media (max-width: 980px) {{ .grid, .metrics, .stack {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Sprint 3 Grounded Medical Report Prototype</h1>
    <div>Scan input -> segmentation -> evidence regions -> grounded report -> hallucination validation</div>
  </header>
  <main>
    <section class="grid">
      <div class="panel image-panel"><h2>Input Scan</h2><img src="original.png" alt="Input MRI scan"></div>
      <div class="panel image-panel"><h2>Predicted Mask</h2><img src="predicted_mask.png" alt="Predicted segmentation mask"></div>
      <div class="panel image-panel"><h2>Evidence Overlay</h2><img src="overlay.png" alt="Evidence overlay with region boxes"></div>
      <div class="panel image-panel"><h2>Probability Heatmap</h2><img src="probability_heatmap.png" alt="Probability heatmap"></div>
    </section>
    <section class="metrics">
      <div class="panel metric">Finding<strong>{'Yes' if findings['finding_present'] else 'No'}</strong></div>
      <div class="panel metric">Mask area<strong>{findings['mask_area_percent']:.2f}%</strong></div>
      <div class="panel metric">Mean probability<strong>{findings['mean_mask_probability']:.2f}</strong></div>
      <div class="panel metric">Evidence regions<strong>{len(regions)}</strong></div>
    </section>
    <section class="stack">
      <div class="panel">
        <h2>Grounded Report</h2>
        <pre>{html.escape(report)}</pre>
      </div>
      <div class="panel">
        <h2>Evidence Regions</h2>
        <table>
          <thead><tr><th>ID</th><th>Area</th><th>Mean Prob.</th><th>Location</th><th>Box</th></tr></thead>
          <tbody>{region_rows}</tbody>
        </table>
        <h2 style="margin-top:20px;">Sentence Validation</h2>
        <table>
          <thead><tr><th>Status</th><th>Report sentence</th><th>Evidence</th><th>Source</th></tr></thead>
          <tbody>{claim_rows}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""
    (output_path / "sprint3_demo.html").write_text(page, encoding="utf-8")


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    image_size = int(checkpoint.get("image_size", args.image_size))

    device = choose_device()
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    original, resized, tensor = load_image(args.image, image_size)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = (prob >= args.pixel_probability_threshold).astype(np.uint8)
    evidence = build_evidence(
        mask,
        prob,
        args.area_threshold,
        args.mean_probability_threshold,
        args.image,
        output_dir,
    )
    display_mask = mask if evidence["mask_derived_findings"]["finding_present"] else np.zeros_like(mask)

    original_output = output_dir / "original.png"
    mask_output = output_dir / "predicted_mask.png"
    overlay_output = output_dir / "overlay.png"
    heatmap_output = output_dir / "probability_heatmap.png"
    evidence_json_output = output_dir / "evidence.json"
    evidence_text_output = output_dir / "evidence.txt"
    panel_output = output_dir / "demo_panel.png"

    original.save(original_output)
    save_mask(display_mask, output_dir / "predicted_mask.png")
    regions = evidence.get("evidence_regions", [])
    overlay = make_overlay(resized, display_mask, regions)
    overlay.save(overlay_output)
    heatmap = make_probability_heatmap(resized, prob, regions)
    heatmap.save(heatmap_output)

    evidence_json_output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    evidence_text_output.write_text(evidence_text(evidence), encoding="utf-8")

    report, validation = write_report_outputs(evidence, output_dir)
    make_demo_panel(resized, Image.fromarray(display_mask * 255), overlay, report, evidence, panel_output)
    write_html_demo(evidence, report, validation, output_dir)

    print(
        "Saved original, mask, overlay, evidence, report, validation, "
        f"and demo panel to {output_dir}"
    )
    print(f"Report validation: {'passed' if validation['is_valid'] else 'failed'}")
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
