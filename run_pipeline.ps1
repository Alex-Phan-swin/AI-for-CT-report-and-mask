# PowerShell script to run the UNet pipeline (training skipped by default)
param(
    [string]$DataDir = "dataset/archive/kaggle_3m",
    [string]$DemoDir = "demo_input",
    [int]$Epochs = 5,
    [int]$BatchSize = 4,
    [int]$ImageSize = 256,
    [switch]$Train,
    [switch]$SkipEvaluation,
    [switch]$SkipDemo,
    [switch]$UseQwen,
    [switch]$ShowPanel,
    [switch]$OpenFolder
)

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "UNET BRAIN TUMOR SEGMENTATION PIPELINE" -ForegroundColor White
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "Training: $(if ($Train) { 'ENABLED' } else { 'SKIPPED' })" -ForegroundColor Yellow
Write-Host "Qwen Formatting: $(if ($UseQwen) { 'ENABLED' } else { 'DISABLED' })" -ForegroundColor Yellow
Write-Host "Demo Directory: $DemoDir" -ForegroundColor Gray
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Change to the main_pipeline directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$mainPipelineDir = Join-Path $scriptDir "main_pipeline"
Set-Location $mainPipelineDir

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
if ($UseQwen) { $args += "--use-qwen" }
if ($ShowPanel) { $args += "--show-panel" }
if ($OpenFolder) { $args += "--open-folder" }

# Run the pipeline
& python $args

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Green
Write-Host "Pipeline execution completed." -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green