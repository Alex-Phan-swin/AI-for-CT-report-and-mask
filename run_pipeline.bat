@echo off
REM Windows batch script to run the UNet pipeline (training skipped by default)
echo Starting UNet Brain Tumor Segmentation Pipeline (training skipped)...
echo To include training, run: python run_pipeline.py --train
echo.

REM Change to the main_pipeline directory
cd /d "%~dp0main_pipeline"

REM Run the pipeline
python run_pipeline.py %*

echo.
echo Pipeline execution completed.
pause