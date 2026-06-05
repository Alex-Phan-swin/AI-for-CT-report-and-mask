# Sprint 3 Alpha Release: Medical AI Segmentation And Grounded Reporting

This alpha release moves beyond the earlier image-level classifier by producing pixel-level localisation of suspicious tumour regions and generating a grounded report from the predicted mask, colour-coded visual evidence, and optional Qwen report generation.

## Sprint 3 Pipeline

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
Sprint3AlphaRelease/dataset/
```

The loader expects image files with matching mask files named like:

```text
TCGA_..._1.tif
TCGA_..._1_mask.tif
```

This matches the common LGG/Kaggle structure.

## Setup

```bash
cd Sprint3AlphaRelease
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The retained dataset path for this local prototype is:

```text
dataset/archive/kaggle_3m
```

## Run The Web Demo With Qwen

From the `Sprint3AlphaRelease` folder, run:

```bash
MEDISCAN_USE_QWEN=1 python3 src/demo_web.py
```

Then open the local website shown in the terminal, usually:

```text
http://127.0.0.1:8765
```

This starts the upload webpage with Qwen enabled for report generation. U-Net still performs the segmentation first, then Qwen writes the report from the structured segmentation evidence. If Qwen is working, the report area shows:

```text
Report writer: Qwen
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
outputs/original.png
outputs/predicted_mask.png
outputs/overlay.png
outputs/evidence.json
outputs/evidence.txt
outputs/report.txt
outputs/report_validation.json
outputs/report_validation.txt
outputs/demo_panel.png
outputs/ground_truth_panel.png
```

## 4. Evaluate The Model

```bash
python src/evaluate.py
```

This writes segmentation and image-level metrics to:

```text
outputs/evaluation_metrics.json
```

## Sprint 3 Framing

Sprint 1 classified scans at image level. Sprint 3 performs segmentation, producing visual evidence that grounds the generated report. The report is constrained by measurable mask outputs such as abnormality presence, approximate mask area, colour-coded evidence regions, and image location.

The current Sprint 3 alpha flow is:

```text
MRI upload -> U-Net segmentation -> colour-coded overlay -> Qwen grounded report -> validation
```
