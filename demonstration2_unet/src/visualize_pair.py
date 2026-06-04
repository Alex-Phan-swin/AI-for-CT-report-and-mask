import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from dataset import BrainTumorSegmentationDataset


def main(args):
    dataset = BrainTumorSegmentationDataset(args.data_dir, image_size=args.image_size)
    sample = dataset[args.index]
    image = sample["image"].squeeze().numpy()
    mask = sample["mask"].squeeze().numpy()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("MRI image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Ground-truth mask")
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(mask, cmap="Reds", alpha=0.45)
    axes[2].set_title("Mask overlay")

    for axis in axes:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f"Saved preview to {output}")
    print(f"Image: {sample['image_path']}")
    print(f"Mask: {sample['mask_path']}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output", default="outputs/sample_pair.png")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
