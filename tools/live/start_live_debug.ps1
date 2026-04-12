[CmdletBinding()]
param(
    [ValidateSet("single_person", "two_player_left_right")]
    [string]$TrackingMode = "single_person",

    [ValidateRange(1, 100000)]
    [int]$PrintEveryN = 10,

    [switch]$NoQuietWarmup,

    [switch]$KeepJson
)

$ErrorActionPreference = "Stop"

function Resolve-ConfigPath {
    param(
        [string]$PathValue,
        [string]$BaseDir
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        return $null
    }

    $expanded = [Environment]::ExpandEnvironmentVariables($PathValue)
    if ([System.IO.Path]::IsPathRooted($expanded)) {
        return [System.IO.Path]::GetFullPath($expanded)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $expanded))
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir "..\.."))
Set-Location $repoRoot

$configPath = Join-Path $repoRoot "configs/local_paths.yaml"
$exampleConfigPath = Join-Path $repoRoot "configs/local_paths.example.yaml"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python was not found on PATH. Activate your venv or install Python first."
}

if (-not (Test-Path -Path $configPath -PathType Leaf)) {
    Write-Host "Missing local runtime config: $configPath" -ForegroundColor Yellow
    Write-Host "Copy and edit the example first:" -ForegroundColor Yellow
    Write-Host "  Copy-Item '$exampleConfigPath' '$configPath'" -ForegroundColor Yellow
    Write-Host "Then set openpose_demo_exe to your local OpenPoseDemo.exe path." -ForegroundColor Yellow
    exit 1
}

$configJson = python -c "import json, pathlib, sys, yaml; p=pathlib.Path(sys.argv[1]); data=yaml.safe_load(p.read_text(encoding='utf-8')) or {}; print(json.dumps(data))" "$configPath"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to parse config file at $configPath"
}

$config = $configJson | ConvertFrom-Json

$openposeExe = Resolve-ConfigPath -PathValue $config.openpose_demo_exe -BaseDir $repoRoot
if (-not $openposeExe) {
    throw "openpose_demo_exe is required in $configPath"
}
if (-not (Test-Path -Path $openposeExe -PathType Leaf)) {
    throw "OpenPose executable not found: $openposeExe"
}

$modelsDir = Resolve-ConfigPath -PathValue $config.openpose_models_dir -BaseDir $repoRoot
if ($modelsDir -and -not (Test-Path -Path $modelsDir -PathType Container)) {
    throw "openpose_models_dir is set but missing: $modelsDir"
}

$liveJsonDir = Resolve-ConfigPath -PathValue $config.live_json_dir -BaseDir $repoRoot
if (-not $liveJsonDir) {
    $liveJsonDir = Join-Path $repoRoot "data/raw/live_buffer/openpose_session/live_test"
}

New-Item -ItemType Directory -Path $liveJsonDir -Force | Out-Null

if (-not $KeepJson) {
    Get-ChildItem -Path $liveJsonDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Remove-Item -Force
}

$pythonModuleCheck = python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('src.inference.live_openpose_debug') else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Could not import src.inference.live_openpose_debug. Run this script from repo root with project dependencies installed."
}

$modelPath = Join-Path $repoRoot "models/checkpoints/best_mlp.keras"
if (-not (Test-Path -Path $modelPath -PathType Leaf)) {
    Write-Host "Warning: model checkpoint not found at $modelPath" -ForegroundColor Yellow
}

$labelMapPath = Join-Path $repoRoot "data/processed/label_map.json"
if (-not (Test-Path -Path $labelMapPath -PathType Leaf)) {
    Write-Host "Warning: label map not found at $labelMapPath" -ForegroundColor Yellow
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$inferenceLogsDir = Join-Path $repoRoot "logs/inference"
New-Item -ItemType Directory -Path $inferenceLogsDir -Force | Out-Null
$logCsvPath = Join-Path $inferenceLogsDir "live_debug_launcher_$timestamp.csv"
$summaryPath = Join-Path $inferenceLogsDir "live_debug_launcher_${timestamp}_summary.json"

$openposeArgs = @("--write_json", $liveJsonDir)
if ($modelsDir) {
    $openposeArgs += @("--model_folder", $modelsDir)
}
if ($null -ne $config.openpose_camera -and "$($config.openpose_camera)" -ne "") {
    $openposeArgs += @("--camera", "$($config.openpose_camera)")
}

Write-Host "Starting OpenPose..." -ForegroundColor Cyan
Write-Host "  EXE: $openposeExe"
Write-Host "  JSON output: $liveJsonDir"

$openposeProcess = Start-Process -FilePath $openposeExe -ArgumentList $openposeArgs -PassThru
Start-Sleep -Milliseconds 800

$classifierArgs = @(
    "-m", "src.inference.live_openpose_debug",
    "--json-dir", $liveJsonDir,
    "--tracking-mode", $TrackingMode,
    "--print-every-n", "$PrintEveryN",
    "--log-csv", $logCsvPath
)
if (-not $NoQuietWarmup) {
    $classifierArgs += "--quiet-warmup"
}

Write-Host "Starting classifier..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C in this terminal to stop both processes." -ForegroundColor Cyan
Write-Host ""

try {
    & python @classifierArgs
}
finally {
    if ($openposeProcess -and -not $openposeProcess.HasExited) {
        Write-Host "Stopping OpenPose (PID $($openposeProcess.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $openposeProcess.Id -Force -ErrorAction SilentlyContinue
    }

    Write-Host ""
    Write-Host "Live session ended." -ForegroundColor Green
    Write-Host "JSON folder: $liveJsonDir"
    Write-Host "Classifier log: $logCsvPath"
    Write-Host "Classifier summary: $summaryPath"
}
