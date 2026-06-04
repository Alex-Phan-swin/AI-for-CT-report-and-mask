import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import BrainTumorSegmentationDataset
from model import UNet


def dice_score_from_logits(logits, masks, threshold=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean()


def dice_loss_from_logits(logits, masks, eps=1e-7):
    probs = torch.sigmoid(logits)
    intersection = (probs * masks).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + eps) / (union + eps)).mean()


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = BrainTumorSegmentationDataset(args.data_dir, image_size=args.image_size)
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = choose_device()
    model = UNet().to(device)
    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Samples: {len(dataset)} | Train: {train_size} | Val: {val_size}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = bce(logits, masks) + dice_loss_from_logits(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)

                loss = bce(logits, masks) + dice_loss_from_logits(logits, masks)
                val_loss += loss.item()
                val_dice += dice_score_from_logits(logits, masks).item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        val_dice /= max(1, len(val_loader))

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_size": args.image_size,
        },
        output_path,
    )
    print(f"Saved model to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output", default="models/unet_brain_mri.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
