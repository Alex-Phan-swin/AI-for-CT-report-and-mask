# PowerShell script to run the UNet pipeline (training skipped by default)
param(
    [string]$DataDir = "dataset/archive/kaggle_3m",
    [string]$DemoDir = "demo_input",
    [int]$Epochs = 5,
    [int]$BatchSize = 4,
    [int]$ImageSize = 256,
    [switch]$Train,  # Changed from SkipTraining
    [switch]$SkipEvaluation,
    [switch]$SkipDemo
)

Write-Host "Starting UNet Brain Tumor Segmentation Pipeline (training skipped by default)..." -ForegroundColor Green
Write-Host "To include training, use -Train switch" -ForegroundColor Yellow
Write-Host "Data directory: $DataDir" -ForegroundColor Cyan
Write-Host "Demo directory: $DemoDir" -ForegroundColor Cyan
Write-Host ""

# Change to the demonstration2_unet directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$demonstrationDir = Join-Path $scriptDir "demonstration2_unet"
Set-Location $demonstrationDir

# Build arguments
$args = @(
    "run_pipeline.py",
    "--data-dir", $DataDir,
    "--demo-dir", $DemoDir,
    "--epochs", $Epochs.ToString(),
    "--batch-size", $BatchSize.ToString(),
    "--image-size", $ImageSize.ToString()
)

if ($Train) { $args += "--train" }
if ($SkipEvaluation) { $args += "--skip-evaluation" }
if ($SkipDemo) { $args += "--skip-demo" }

# Run the pipeline
& python $args

Write-Host ""
Write-Host "Pipeline execution completed." -ForegroundColor Green