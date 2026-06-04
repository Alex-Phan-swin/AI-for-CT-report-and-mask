import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from model import UNet
from qwen_report_generator import MedicalImageReportGenerator


# =========================
# DEVICE
# =========================
def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================
# LOAD IMAGE
# =========================
def load_image(image_path, size=256):
    img = Image.open(image_path).convert("L")
    img = img.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor


# =========================
# RUN INFERENCE
# =========================
def predict(model, image_tensor, device, threshold=0.5):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()

    return mask.squeeze().cpu().numpy(), probs.squeeze().cpu().numpy()


# =========================
# MAIN
# =========================
def main(args):
    device = choose_device()

    # -------------------------
    # LOAD MODEL
    # -------------------------
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # -------------------------
    # LOAD IMAGE
    # -------------------------
    image_tensor = load_image(args.image, size=256)

    # -------------------------
    # PREDICT
    # -------------------------
    pred_mask, prob_map = predict(model, image_tensor, device)

    # -------------------------
    # SAVE OUTPUTS
    # -------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save prediction mask (THIS WAS MISSING BEFORE)
    mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
    mask_path = output_dir / "pred_mask.png"
    mask_img.save(mask_path)

    # Save probability map (optional but useful)
    prob_img = Image.fromarray((prob_map * 255).astype(np.uint8))
    prob_path = output_dir / "prob_map.png"
    prob_img.save(prob_path)

    print(f"Saved prediction mask → {mask_path}")
    print(f"Saved probability map → {prob_path}")

    # -------------------------
    # GENERATE REPORT
    # -------------------------
    generator = MedicalImageReportGenerator()

    report = generator.generate_report(
        image_path=args.image,
        modality=args.modality,
        max_new_tokens=256,
    )

    report_path = output_dir / "report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved report → {report_path}")


# =========================
# ARGS
# =========================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--modality", type=str, default="auto")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())