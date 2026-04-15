[CmdletBinding()]
param(
    [ValidateSet("single_person", "two_player_left_right")]
    [string]$TrackingMode = "single_person",

    [ValidateRange(1, 100000)]
    [int]$PrintEveryN = 10,

    [switch]$NoQuietWarmup,

    [ValidateRange(0.0, 1.0)]
    [double]$AcceptThreshold = 0.80,

    [ValidateRange(0.0, 1.0)]
    [double]$MarginThreshold = 0.20,

    [ValidateSet("terminal", "none")]
    [string]$OverlayMode = "terminal",

    [double]$SideSplitX = [double]::NaN,

    [switch]$KeepJson,

    [ValidateRange(0, 600)]
    [int]$OpenPoseStartupTimeoutSec = 60,

    [switch]$KillOpenPoseOnExit
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

$openposeWorkingDir = Resolve-ConfigPath -PathValue $config.openpose_working_dir -BaseDir $repoRoot
if (-not $openposeWorkingDir) {
    $openposeWorkingDir = Split-Path -Parent $openposeExe
}
if (-not (Test-Path -Path $openposeWorkingDir -PathType Container)) {
    throw "OpenPose working directory not found: $openposeWorkingDir"
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
$preexistingJsonCount = (Get-ChildItem -Path $liveJsonDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Measure-Object).Count

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
Write-Host "  OpenPose EXE: $openposeExe"
Write-Host "  OpenPose working dir: $openposeWorkingDir"
if ($modelsDir) {
    Write-Host "  OpenPose model folder: $modelsDir"
}
else {
    Write-Host "  OpenPose model folder: <not set>"
}
Write-Host "  Live JSON dir: $liveJsonDir"
Write-Host "  Live JSON pre-launch file count: $preexistingJsonCount"

$openposeProcess = Start-Process -FilePath $openposeExe -ArgumentList $openposeArgs -WorkingDirectory $openposeWorkingDir -PassThru

$firstJsonDetected = $false
$detectedJsonPath = $null
$timeoutDisabled = $OpenPoseStartupTimeoutSec -eq 0
if ($timeoutDisabled) {
    Write-Host "Waiting for first OpenPose JSON frame in $liveJsonDir (timeout disabled)." -ForegroundColor Cyan
}
else {
    $deadline = (Get-Date).AddSeconds($OpenPoseStartupTimeoutSec)
    Write-Host "Waiting for first OpenPose JSON frame in $liveJsonDir (timeout: ${OpenPoseStartupTimeoutSec}s)..." -ForegroundColor Cyan
}

while ($true) {
    if ((-not $timeoutDisabled) -and ((Get-Date) -ge $deadline)) {
        break
    }

    if ($openposeProcess.HasExited) {
        throw "OpenPose exited early (PID $($openposeProcess.Id), exit code $($openposeProcess.ExitCode)) before producing JSON frames."
    }

    $firstJson = Get-ChildItem -Path $liveJsonDir -Filter "*.json" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime |
        Select-Object -First 1

    if ($firstJson) {
        $firstJsonDetected = $true
        $detectedJsonPath = $firstJson.FullName
        break
    }

    Start-Sleep -Milliseconds 250
}

if (-not $firstJsonDetected) {
    $jsonCount = (Get-ChildItem -Path $liveJsonDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host ""
    Write-Host "WARNING: No OpenPose JSON detected before startup timeout." -ForegroundColor Yellow
    Write-Host "  OpenPose EXE: $openposeExe" -ForegroundColor Yellow
    Write-Host "  OpenPose working dir: $openposeWorkingDir" -ForegroundColor Yellow
    if ($modelsDir) {
        Write-Host "  OpenPose model folder: $modelsDir" -ForegroundColor Yellow
    }
    else {
        Write-Host "  OpenPose model folder: <not set>" -ForegroundColor Yellow
    }
    Write-Host "  Live JSON dir: $liveJsonDir" -ForegroundColor Yellow
    Write-Host "  JSON files detected: $jsonCount" -ForegroundColor Yellow
    Write-Host "OpenPose was intentionally left running for manual inspection." -ForegroundColor Yellow
    Write-Host "Close the OpenPose window/process manually when you are done debugging." -ForegroundColor Yellow
    exit 1
}

Write-Host "First OpenPose JSON detected: $detectedJsonPath" -ForegroundColor Green

$classifierArgs = @(
    "-m", "src.inference.live_openpose_debug",
    "--json-dir", $liveJsonDir,
    "--tracking-mode", $TrackingMode,
    "--accept-threshold", "$AcceptThreshold",
    "--margin-threshold", "$MarginThreshold",
    "--overlay-mode", $OverlayMode,
    "--print-every-n", "$PrintEveryN",
    "--log-csv", $logCsvPath
)

if ($TrackingMode -eq "two_player_left_right") {
    $resolvedSideSplitX = $null
    if (-not [double]::IsNaN($SideSplitX)) {
        $resolvedSideSplitX = $SideSplitX
        Write-Host ("Using explicit side split x: {0}" -f $resolvedSideSplitX) -ForegroundColor Cyan
    }
    else {
        $cameraSelector = "0"
        if ($null -ne $config.openpose_camera -and "$($config.openpose_camera)" -ne "") {
            $cameraSelector = "$($config.openpose_camera)"
        }
        $cameraProbeJson = python -c "import cv2, json, sys; cam = cv2.VideoCapture(sys.argv[1]); width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)); cam.release(); print(json.dumps({'width': width, 'height': height}))" "$cameraSelector"
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($cameraProbeJson)) {
            $cameraProbe = $cameraProbeJson | ConvertFrom-Json
            if ($cameraProbe.width -gt 0) {
                $resolvedSideSplitX = [double]$cameraProbe.width / 2.0
                Write-Host ("Auto side split x from camera width {0}: {1}" -f $cameraProbe.width, $resolvedSideSplitX) -ForegroundColor Cyan
            }
        }
        if ($null -eq $resolvedSideSplitX) {
            Write-Host "Warning: unable to auto-detect camera width; classifier will use dynamic split midpoint." -ForegroundColor Yellow
        }
    }
    if ($null -ne $resolvedSideSplitX) {
        $classifierArgs += @("--side-split-x", "$resolvedSideSplitX")
    }
}
if (-not $NoQuietWarmup) {
    $classifierArgs += "--quiet-warmup"
}

Write-Host "Starting classifier..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C in this terminal to stop the classifier." -ForegroundColor Cyan
if ($KillOpenPoseOnExit) {
    Write-Host "OpenPose auto-cleanup is ENABLED for this run (-KillOpenPoseOnExit)." -ForegroundColor Yellow
}
else {
    Write-Host "OpenPose auto-cleanup is DISABLED by default for machine safety; close OpenPose manually when done." -ForegroundColor Yellow
}
Write-Host ""

try {
    & python @classifierArgs
}
finally {
    if ($openposeProcess -and -not $openposeProcess.HasExited) {
        if ($KillOpenPoseOnExit) {
            Write-Host "Stopping OpenPose (PID $($openposeProcess.Id)) because -KillOpenPoseOnExit was set..." -ForegroundColor Yellow
            Stop-Process -Id $openposeProcess.Id -Force -ErrorAction SilentlyContinue
        }
        else {
            Write-Host "OpenPose is still running (PID $($openposeProcess.Id)). Close it manually when finished." -ForegroundColor Yellow
        }
    }

    Write-Host ""
    Write-Host "Live session ended." -ForegroundColor Green
    Write-Host "JSON folder: $liveJsonDir"
    Write-Host "Classifier log: $logCsvPath"
    Write-Host "Classifier summary: $summaryPath"
}
