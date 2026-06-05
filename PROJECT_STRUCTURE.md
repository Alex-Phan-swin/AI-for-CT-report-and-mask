# Project Structure

This repository is organised by sprint so the Sprint 3 alpha release is easy to find and run.

## Main folders

- `Sprint3AlphaRelease/` - Current Sprint 3 alpha release. This is the active web app for MRI segmentation, colour-coded visual evidence, Qwen report generation, and validation.
- `Sprint1/` - Earlier image classification work, liver datasets, liver model checkpoints, and preprocessing/training scripts.
- `Sprint2/` - Earlier VLM and encoder experiments. These are preserved for project history but are not used by the current Sprint 3 alpha web app.

## Sprint 3 key files

- `Sprint3AlphaRelease/src/demo_web.py` - Starts the upload webpage. The HTML, CSS, and JavaScript are embedded in this Python file.
- `Sprint3AlphaRelease/src/predict_report.py` - Runs U-Net segmentation, creates masks, overlays, heatmaps, colour-coded evidence regions, and metrics.
- `Sprint3AlphaRelease/src/report_generator.py` - Creates the grounded report from visual evidence and optionally uses Qwen.
- `Sprint3AlphaRelease/models/unet_brain_mri.pth` - Sprint 3 U-Net model checkpoint used by the website.
- `Sprint3AlphaRelease/demo_input/` - Small demo MRI image used by the "Use demo MRI" button.

## Run the Sprint 3 alpha website

Open a terminal and run:

```bash
cd "/Users/christomalappadan/Documents/UNI/2026 Semester 1/COS40005 Computing Technology Project A/COS40005-Computing-Technology-Project-A-H/Sprint3AlphaRelease"
```

Then run with Qwen enabled:

```bash
MEDISCAN_USE_QWEN=1 python3 src/demo_web.py
```

The website should open at:

```text
http://127.0.0.1:8765
```

## Is Sprint 3 using VLM or Encoder?

No. The current Sprint 3 alpha release does not import or run the old `VLM/` or `Encoder/` folders.

Sprint 3 uses:

```text
U-Net segmentation -> colour-coded evidence regions -> Qwen report generation -> validation
```

The old VLM and Encoder work is stored under `Sprint2/` for reference only.

## Optional Qwen report generation

By default, the demo can use the deterministic grounded report generator. To use Qwen as the report-writing layer, install the optional language-model dependencies and enable it with an environment variable.

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
