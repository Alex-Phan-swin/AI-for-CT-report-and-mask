# Project Structure

This repository currently contains older liver classification work plus the Sprint 3 U-Net web demo.

## Main folders

- `demonstration2_unet/` - Sprint 3 MRI segmentation web demo.
- `demonstration2_unet/src/demo_web.py` - Starts the upload webpage. The HTML, CSS, and JavaScript are embedded in this Python file.
- `demonstration2_unet/src/predict_report.py` - Runs U-Net segmentation, creates masks, overlays, heatmaps, colour-coded evidence regions, and metrics.
- `demonstration2_unet/src/report_generator.py` - Creates the grounded report from the visual evidence.
- `demonstration2_unet/models/unet_brain_mri.pth` - Sprint 3 U-Net model checkpoint used by the website.
- `demonstration2_unet/demo_input/` - Small demo MRI image used by the "Use demo MRI" button.
- `scripts/` - Earlier liver image preprocessing/training/prediction scripts.
- `VLM/` - Earlier report-generation or vision-language model experiments.
- `Dataset/` and `images_dataset/` - Dataset folders used by earlier work.
- `models/` and `backup_models/` - Earlier liver classifier model checkpoints.
- `Encoder/` - Earlier encoder/feature extraction work.

## Run the Sprint 3 website

Open a terminal and run:

```bash
cd "/Users/christomalappadan/Documents/UNI/2026 Semester 1/COS40005 Computing Technology Project A/COS40005-Computing-Technology-Project-A-H/demonstration2_unet"
```

Then run:

```bash
python3 src/demo_web.py
```

The website should open at:

```text
http://127.0.0.1:8765
```

## Optional Qwen report generation

By default, the demo uses the deterministic grounded report generator. To use Qwen as the report-writing layer, install the optional language-model dependencies and enable it with an environment variable.

Qwen is used after U-Net segmentation. It receives only the structured evidence, mask statistics, overlay paths, colour-coded regions, and baseline report. The validation layer still blocks unsupported claims such as tumour type, grade, prognosis, treatment, malignancy, or diagnosis.

Install dependencies:

```bash
pip3 install -r requirements.txt
```

Run with Qwen enabled:

```bash
MEDISCAN_USE_QWEN=1 python3 src/demo_web.py
```

Use a different Qwen model if needed:

```bash
MEDISCAN_USE_QWEN=1 MEDISCAN_QWEN_MODEL="Qwen/Qwen2.5-0.5B-Instruct" python3 src/demo_web.py
```
