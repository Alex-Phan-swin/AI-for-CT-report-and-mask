import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import BrainTumorSegmentationDataset
from model import UNet


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def safe_divide(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def evaluate(args):
    dataset = BrainTumorSegmentationDataset(args.data_dir, image_size=args.image_size)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = choose_device()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pixel_tp = 0.0
    pixel_fp = 0.0
    pixel_tn = 0.0
    pixel_fn = 0.0
    image_tp = 0
    image_fp = 0
    image_tn = 0
    image_fn = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            probs = torch.sigmoid(model(images))
            preds = (probs >= args.pixel_probability_threshold).float()

            pixel_tp += float((preds * masks).sum().item())
            pixel_fp += float((preds * (1 - masks)).sum().item())
            pixel_tn += float(((1 - preds) * (1 - masks)).sum().item())
            pixel_fn += float(((1 - preds) * masks).sum().item())

            pred_area = preds.mean(dim=(1, 2, 3)) * 100
            masked_probs = probs * preds
            pred_pixels = preds.sum(dim=(1, 2, 3))
            mean_prob = masked_probs.sum(dim=(1, 2, 3)) / pred_pixels.clamp_min(1.0)
            pred_present = (pred_area >= args.area_threshold) & (
                mean_prob >= args.mean_probability_threshold
            )
            true_present = masks.sum(dim=(1, 2, 3)) > 0

            image_tp += int((pred_present & true_present).sum().item())
            image_fp += int((pred_present & ~true_present).sum().item())
            image_tn += int((~pred_present & ~true_present).sum().item())
            image_fn += int((~pred_present & true_present).sum().item())

    dice = safe_divide(2 * pixel_tp, 2 * pixel_tp + pixel_fp + pixel_fn)
    iou = safe_divide(pixel_tp, pixel_tp + pixel_fp + pixel_fn)
    precision = safe_divide(pixel_tp, pixel_tp + pixel_fp)
    recall = safe_divide(pixel_tp, pixel_tp + pixel_fn)
    pixel_accuracy = safe_divide(pixel_tp + pixel_tn, pixel_tp + pixel_tn + pixel_fp + pixel_fn)

    image_total = image_tp + image_fp + image_tn + image_fn
    image_accuracy = safe_divide(image_tp + image_tn, image_total)
    image_precision = safe_divide(image_tp, image_tp + image_fp)
    image_recall = safe_divide(image_tp, image_tp + image_fn)

    metrics = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "validation_samples": len(val_dataset),
        "pixel_metrics": {
            "dice": round(dice, 4),
            "iou": round(iou, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "pixel_accuracy": round(pixel_accuracy, 4),
        },
        "image_level_metrics": {
            "accuracy": round(image_accuracy, 4),
            "precision": round(image_precision, 4),
            "recall": round(image_recall, 4),
            "true_positive": image_tp,
            "false_positive": image_fp,
            "true_negative": image_tn,
            "false_negative": image_fn,
        },
        "decision_thresholds": {
            "pixel_probability_threshold": args.pixel_probability_threshold,
            "area_threshold_percent": args.area_threshold,
            "mean_probability_threshold": args.mean_probability_threshold,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/unet_brain_mri.pth")
    parser.add_argument("--data-dir", default="dataset/archive/kaggle_3m")
    parser.add_argument("--output", default="outputs/evaluation_metrics.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pixel-probability-threshold", type=float, default=0.5)
    parser.add_argument("--area-threshold", type=float, default=1.0)
    parser.add_argument("--mean-probability-threshold", type=float, default=0.8)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
