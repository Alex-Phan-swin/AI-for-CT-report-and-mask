#!/usr/bin/env python3
"""
Master pipeline script to run all UNet brain tumor segmentation components.
This script runs the complete workflow: data prep → training → evaluation → prediction.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    try:
        result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run complete UNet brain tumor segmentation pipeline")
    parser.add_argument("--data-dir", default="dataset/archive/kaggle_3m",
                       help="Path to training data directory")
    parser.add_argument("--demo-dir", default="demo_input",
                       help="Path to demo input directory for prediction")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--image-size", type=int, default=256,
                       help="Image size for training")
    parser.add_argument("--train", action="store_true",
                       help="Run training (skipped by default to save time)")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation step")
    parser.add_argument("--skip-demo", action="store_true",
                       help="Skip demo folder analysis")

    args = parser.parse_args()

    # Define paths
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src"
    models_dir = base_dir / "models"
    outputs_dir = base_dir / "outputs"

    # Ensure directories exist
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    # Pipeline steps
    steps = []

    # Step 1: Data preprocessing (separate_tumour_folders.py)
    steps.append({
        "cmd": [sys.executable, str(src_dir / "separate_tumour_folders.py")],
        "desc": "Data preprocessing - organizing tumor folders"
    })

    # Step 2: Training (train.py) - only run if explicitly requested
    if args.train:
        steps.append({
            "cmd": [
                sys.executable, str(src_dir / "train.py"),
                "--data-dir", args.data_dir,
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--image-size", str(args.image_size)
            ],
            "desc": f"Training UNet model ({args.epochs} epochs)"
        })

    # Step 3: Evaluation (evaluate.py) - skip if requested
    if not args.skip_evaluation:
        steps.append({
            "cmd": [
                sys.executable, str(src_dir / "evaluate.py"),
                "--data-dir", args.data_dir,
                "--image-size", str(args.image_size)
            ],
            "desc": "Evaluating trained model"
        })

    # Step 4: Demo folder analysis (analyse_demo_folder.py) - skip if requested
    if not args.skip_demo:
        steps.append({
            "cmd": [
                sys.executable, str(src_dir / "analyse_demo_folder.py"),
                "--input-dir", args.demo_dir
            ],
            "desc": "Analyzing demo input folder"
        })

    # Run all steps
    print("Starting UNet Brain Tumor Segmentation Pipeline")
    print(f"Data directory: {args.data_dir}")
    print(f"Demo directory: {args.demo_dir}")
    print(f"Training: {'ENABLED' if args.train else 'SKIPPED (use --train to enable)'}")
    print(f"Total steps: {len(steps)}")

    success_count = 0
    for step in steps:
        if run_command(step["cmd"], step["desc"]):
            success_count += 1
        else:
            print(f"\nPipeline stopped due to failure in: {step['desc']}")
            break

    print(f"\n{'='*60}")
    print(f"Pipeline completed: {success_count}/{len(steps)} steps successful")

    if success_count == len(steps):
        print("✓ All pipeline steps completed successfully!")
        if not args.train:
            print("\nNote: Training was skipped. To train a new model, run with --train")
        print("\nNext steps:")
        print("1. Check outputs/ for evaluation metrics and predictions")
        print("2. Check models/ for trained model checkpoints")
        print("3. Use predict_report.py for individual image predictions:")
        print("   python src/predict_report.py --checkpoint models/unet_brain_mri.pth --image path/to/image.tif")
    else:
        print("✗ Pipeline completed with errors. Check output above for details.")


if __name__ == "__main__":
    main()



# Customization options:

# python run_pipeline.py --epochs 10 --batch-size 8 --data-dir dataset/archive/kaggle_3m

#Skip options:

# Training is skipped by default to save time. To include training, use:
#python run_pipeline.py --train --epochs (Any Number) --batch-size (Any Number)

# Skip evaluation 
#python run_pipeline.py --skip-evaluation

# Skip demo analysis
#python run_pipeline.py --skip-demo