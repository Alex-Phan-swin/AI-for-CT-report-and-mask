# Demonstration 2: U-Net Medical Image Segmentation Prototype

This prototype moves beyond the Sprint 1 classifier by producing pixel-level localisation of suspicious tumour regions and generating a simple report grounded in the predicted mask.

## Demo Pipeline

```text
MRI image
-> U-Net segmentation model
-> predicted tumour mask
-> visual overlay / bounding box
-> mask-derived findings
-> grounded report
```

## Dataset

Recommended dataset: LGG Brain MRI Segmentation from Kaggle.

Place the downloaded/extracted dataset under:

```text
main_pipeline/dataset/
```

The loader expects image files with matching mask files named like:

```text
TCGA_..._1.tif
TCGA_..._1_mask.tif
```

This matches the common LGG/Kaggle structure.

## Setup

```bash
cd main_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The retained dataset path for this local prototype is:

```text
dataset/archive/kaggle_3m
```

## 1. Check Dataset Loading

```bash
python src/visualize_pair.py --data-dir dataset/archive/kaggle_3m --output outputs/sample_pair.png
```

This creates a side-by-side image/mask preview.

## 2. Train U-Net

Start small for a quick demo:

```bash
python src/train.py --data-dir dataset/archive/kaggle_3m --epochs 5 --batch-size 4 --image-size 256
```

The trained model is saved to:

```text
models/unet_brain_mri.pth
```

## 3. Run Inference And Generate Grounded Report

Option A: place exactly one image inside:

```text
demo_input/
```

Then run:

```bash
python src/analyse_demo_folder.py
```

To auto-open the generated demo panel:

```bash
python src/analyse_demo_folder.py --open
```

Option B: analyse a specific image path:

```bash
python src/predict_report.py \
  --checkpoint models/unet_brain_mri.pth \
  --image path/to/demo_image.tif \
  --output-dir outputs
```

Outputs:

```text
outputs/demo/original.png
outputs/demo/predicted_mask.png
outputs/demo/overlay.png
outputs/demo/evidence.json
outputs/demo/evidence.txt
outputs/demo/report.txt
outputs/demo/demo_panel.png
outputs/demo/ground_truth_panel.png
```

## 4. Evaluate The Model

```bash
python src/evaluate.py
```

This writes segmentation and image-level metrics to:

```text
outputs/evaluation/evaluation_metrics.json
```

## Demo Framing

Sprint 1 classified scans at image level. Demonstration 2 performs segmentation, producing visual evidence that grounds the generated report. The report is constrained by measurable mask outputs such as abnormality presence, approximate mask area, and image region.

See `DEMO_NOTES.md` for the client-aligned explanation, limitations, and next steps.
See `EVALUATION_SUMMARY.md` for the latest validation metrics and how to explain them.
