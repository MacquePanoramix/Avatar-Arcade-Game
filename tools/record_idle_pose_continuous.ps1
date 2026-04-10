<#
.SYNOPSIS
Beginner-friendly OpenPose helper for continuous idle-pose sample capture.

.DESCRIPTION
This script launches OpenPose ONE TIME and keeps it running while you capture
multiple idle-pose samples in a row.

OpenPose writes JSON files continuously into one live buffer folder. Each time
you press ENTER, the script copies exactly N NEW JSON files (default 90) from
the live buffer into the next available dataset take folder for the fixed label
`idle`.

Key design goals:
- stable continuous capture (no repeated launch/force-close per sample)
- fixed-size per-sample frame count for cleaner downstream processing
- beginner-friendly prompts and safety checks
#>

# ============================================================================
# EDITABLE DEFAULTS
# ----------------------------------------------------------------------------
# Update these two paths for your machine.
# OpenPose is expected outside this repository.
# ============================================================================
$OpenPoseRoot = "D:\Programs\OpenPose\openpose"
$ProjectRoot = "D:\Documentos\Python Projects\Avatar-Arcade-Game"

$DefaultPerson = "luis"
$DefaultSession = "s01"
$DefaultUseReviewVideo = $true
$DefaultFramesPerTake = 90
$CapturePollIntervalMs = 200
$StartupTimeoutSeconds = 20
$LiveBufferRoot = Join-Path $ProjectRoot "data\\raw\\live_buffer"
$LiveSessionFolderName = "openpose_session"
$IdleGestureLabel = "idle"

# ============================================================================
# Helper: read input with default.
# ============================================================================
function Read-Value {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prompt,

        [Parameter(Mandatory = $false)]
        [string]$Default = ""
    )

    if ([string]::IsNullOrWhiteSpace($Default)) {
        return Read-Host $Prompt
    }

    $value = Read-Host "$Prompt [$Default]"
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $Default
    }

    return $value
}

# ============================================================================
# Helper: convert yes/no input into a boolean.
# ============================================================================
function Read-YesNo {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Prompt,

        [Parameter(Mandatory = $true)]
        [bool]$DefaultValue
    )

    while ($true) {
        $defaultLabel = if ($DefaultValue) { "y" } else { "n" }
        $raw = (Read-Host "$Prompt [$defaultLabel]").Trim().ToLowerInvariant()

        if ([string]::IsNullOrWhiteSpace($raw)) {
            return $DefaultValue
        }

        if ($raw -in @("y", "yes")) {
            return $true
        }

        if ($raw -in @("n", "no")) {
            return $false
        }

        Write-Host "Please type y or n." -ForegroundColor Yellow
    }
}

# ============================================================================
# Helper: extract numeric frame index from OpenPose JSON filename.
# If parse fails, caller can fall back to sequence index boundaries.
# ============================================================================
function Get-FrameIndexFromJsonName {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FileName
    )

    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($FileName)
    $match = [regex]::Match($baseName, '(\d+)(?!.*\d)')
    if ($match.Success) {
        return [int]$match.Groups[1].Value
    }

    return $null
}

# ============================================================================
# Helper: wait for first JSON file in live folder to confirm capture is live.
# ============================================================================
function Wait-ForLiveCaptureStart {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$OpenPoseProcess,

        [Parameter(Mandatory = $true)]
        [string]$LiveRunFolder,

        [Parameter(Mandatory = $true)]
        [int]$PollIntervalMs,

        [Parameter(Mandatory = $true)]
        [int]$TimeoutSeconds
    )

    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    Write-Host ("[DEBUG] Waiting for first JSON in live run folder: {0}" -f $LiveRunFolder) -ForegroundColor DarkGray

    while ($stopwatch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if ($OpenPoseProcess.HasExited) {
            throw "OpenPose exited before capture became live."
        }

        $allJsonFiles = Get-ChildItem -LiteralPath $LiveRunFolder -Filter "*.json" -File -ErrorAction SilentlyContinue
        if ($allJsonFiles.Count -gt 0) {
            Write-Host ("[DEBUG] First JSON detected. Current JSON count: {0}" -f $allJsonFiles.Count) -ForegroundColor DarkGray
            return
        }

        Start-Sleep -Milliseconds $PollIntervalMs
    }

    $filesAfterTimeout = Get-ChildItem -LiteralPath $LiveRunFolder -Filter "*.json" -File -ErrorAction SilentlyContinue
    [int]$jsonCountAfterTimeout = $filesAfterTimeout.Count
    throw ("Startup timed out after {0} seconds. Live run folder: {1}. JSON count: {2}." -f $TimeoutSeconds, $LiveRunFolder, $jsonCountAfterTimeout)
}

# ============================================================================
# Helper: find next available take_### under idle/<person>/<session>.
#
# Rule:
# - start from highest observed take number + 1 (or 1 if none exist)
# - if that folder exists AND already contains JSON files, keep incrementing
# - return first safe take label
# ============================================================================
function Get-NextAvailableIdleTakeLabel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SessionRoot
    )

    # Ensure session root exists so later logic has a stable home.
    New-Item -ItemType Directory -Path $SessionRoot -Force | Out-Null

    $takeNumbers = New-Object System.Collections.Generic.List[int]
    $takeDirs = Get-ChildItem -LiteralPath $SessionRoot -Directory -ErrorAction SilentlyContinue

    foreach ($takeDir in $takeDirs) {
        if ($takeDir.Name -match '^take_(\d+)$') {
            $takeNumbers.Add([int]$matches[1])
        }
    }

    $candidateNumber = 1
    if ($takeNumbers.Count -gt 0) {
        $maxTake = [int](($takeNumbers | Measure-Object -Maximum).Maximum)
        $candidateNumber = $maxTake + 1
    }

    while ($true) {
        $candidateLabel = "take_" + $candidateNumber.ToString("000")
        $candidatePath = Join-Path $SessionRoot $candidateLabel

        if (-not (Test-Path -LiteralPath $candidatePath -PathType Container)) {
            return $candidateLabel
        }

        $existingJson = Get-ChildItem -LiteralPath $candidatePath -Filter "*.json" -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $existingJson) {
            # Folder exists but does not contain JSON yet, so it is safe to use.
            return $candidateLabel
        }

        # Folder already has JSON; skip forward to protect existing data.
        $candidateNumber++
    }
}

Write-Host ""
Write-Host "=== OpenPose Continuous Idle Pose Helper ===" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# 1) Validate static paths.
# ============================================================================
if (-not (Test-Path -LiteralPath $OpenPoseRoot -PathType Container)) {
    Write-Error "OpenPose root not found: $OpenPoseRoot"
    exit 1
}

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    Write-Error "Project root not found: $ProjectRoot"
    exit 1
}

$OpenPoseExe = Join-Path $OpenPoseRoot "bin\\OpenPoseDemo.exe"
if (-not (Test-Path -LiteralPath $OpenPoseExe -PathType Leaf)) {
    Write-Error "OpenPoseDemo.exe not found: $OpenPoseExe"
    exit 1
}

# ============================================================================
# 2) Prompt once for person/session/frames-per-take/video.
# ============================================================================
$person = (Read-Value -Prompt "Person" -Default $DefaultPerson).Trim()
if ([string]::IsNullOrWhiteSpace($person)) {
    $person = $DefaultPerson
}

$session = (Read-Value -Prompt "Session" -Default $DefaultSession).Trim()
if ([string]::IsNullOrWhiteSpace($session)) {
    $session = $DefaultSession
}

$framesInput = (Read-Value -Prompt "Frames per take" -Default "$DefaultFramesPerTake").Trim()
[int]$framesPerTakeResolved = 0
if (-not [int]::TryParse($framesInput, [ref]$framesPerTakeResolved) -or $framesPerTakeResolved -le 0) {
    Write-Error "Frames per take must be a positive integer."
    exit 1
}

[bool]$useReviewVideo = Read-YesNo -Prompt "Save one continuous review video for this idle session? (y/n, default yes)" -DefaultValue $DefaultUseReviewVideo

# ============================================================================
# 3) Build dataset/session paths and resolve first take label.
# ============================================================================
$idleSessionRoot = Join-Path $ProjectRoot ("data\\raw\\openpose_json\\{0}\\{1}\\{2}" -f $IdleGestureLabel, $person, $session)
New-Item -ItemType Directory -Path $idleSessionRoot -Force | Out-Null

$nextTakeLabel = Get-NextAvailableIdleTakeLabel -SessionRoot $idleSessionRoot

Write-Host ""
Write-Host ("Gesture label is fixed to: {0}" -f $IdleGestureLabel) -ForegroundColor Yellow
Write-Host ("Frames per take: {0}" -f $framesPerTakeResolved) -ForegroundColor Yellow
Write-Host ("First available take: {0}" -f $nextTakeLabel) -ForegroundColor Yellow
Write-Host ""

# ============================================================================
# 4) Prepare live buffer folder and unique run folder.
# ============================================================================
$liveSessionFolder = Join-Path $LiveBufferRoot $LiveSessionFolderName
New-Item -ItemType Directory -Path $liveSessionFolder -Force | Out-Null

$runTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$liveRunFolder = Join-Path $liveSessionFolder ("session_{0}" -f $runTimestamp)
New-Item -ItemType Directory -Path $liveRunFolder -Force | Out-Null

$manifestCsvPath = Join-Path $liveRunFolder "idle_manifest.csv"

$reviewVideoPath = ""
if ($useReviewVideo) {
    $reviewVideoPath = Join-Path $liveRunFolder "session_review.avi"
}

$preexistingJson = Get-ChildItem -LiteralPath $liveRunFolder -Filter "*.json" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($preexistingJson) {
    Write-Error "Live run folder already contains JSON files: $liveRunFolder"
    exit 1
}

# ============================================================================
# 5) Launch OpenPose ONCE with JSON output pointed to live run folder.
# ============================================================================
$openPoseArgParts = @(
    "--number_people_max 1",
    "--tracking 1",
    "--write_json `"$liveRunFolder`""
)

if ($useReviewVideo) {
    $openPoseArgParts += "--write_video `"$reviewVideoPath`""
}

$openPoseArgs = ($openPoseArgParts -join " ")
$openPoseProcess = $null
$completedSampleCount = 0

try {
    Write-Host "Launching OpenPose once for continuous idle capture..." -ForegroundColor Cyan
    Write-Host ("OpenPose arguments: {0}" -f $openPoseArgs) -ForegroundColor DarkGray

    $openPoseProcess = Start-Process -FilePath $OpenPoseExe -ArgumentList $openPoseArgs -WorkingDirectory $OpenPoseRoot -PassThru
    Write-Host ("OpenPose started (PID: {0})." -f $openPoseProcess.Id) -ForegroundColor Green

    Write-Host ("Waiting for live JSON output (timeout: {0}s)..." -f $StartupTimeoutSeconds) -ForegroundColor Yellow
    Wait-ForLiveCaptureStart -OpenPoseProcess $openPoseProcess -LiveRunFolder $liveRunFolder -PollIntervalMs $CapturePollIntervalMs -TimeoutSeconds $StartupTimeoutSeconds
    Write-Host "Capture is live." -ForegroundColor Green

    # ========================================================================
    # 6) Repeated idle sample loop.
    # - ENTER => capture exactly N new JSON frames for one idle sample
    # - q     => end session cleanly
    # ========================================================================
    while ($true) {
        # Always compute and print the next safe take label before each sample.
        $nextTakeLabel = Get-NextAvailableIdleTakeLabel -SessionRoot $idleSessionRoot
        Write-Host ""
        Write-Host ("Next idle take label: {0}" -f $nextTakeLabel) -ForegroundColor Cyan

        $command = (Read-Host "Press ENTER to record one idle sample, or type q to quit.").Trim().ToLowerInvariant()
        if ($command -eq "q") {
            Write-Host "Idle capture session ended by user request." -ForegroundColor Yellow
            break
        }

        $destJsonDir = Join-Path $idleSessionRoot $nextTakeLabel
        New-Item -ItemType Directory -Path $destJsonDir -Force | Out-Null

        # Safety: if destination already has JSON, skip ahead without overwriting.
        $existingDestJson = Get-ChildItem -LiteralPath $destJsonDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($existingDestJson) {
            Write-Host ("Destination already has JSON, skipping: {0}" -f $destJsonDir) -ForegroundColor Yellow
            continue
        }

        # Baseline count before this sample starts. We only copy files that
        # arrive AFTER this baseline to guarantee exactly NEW frames.
        $baselineFiles = Get-ChildItem -LiteralPath $liveRunFolder -Filter "*.json" -File -ErrorAction SilentlyContinue |
            Sort-Object Name
        $baselineCount = $baselineFiles.Count

        Write-Host ("Capturing exactly {0} NEW idle frames..." -f $framesPerTakeResolved) -ForegroundColor Yellow

        $selectedTakeFiles = $null
        while ($true) {
            if ($openPoseProcess.HasExited) {
                throw "OpenPose exited unexpectedly during idle sample capture."
            }

            $allFilesNow = Get-ChildItem -LiteralPath $liveRunFolder -Filter "*.json" -File -ErrorAction SilentlyContinue |
                Sort-Object Name

            $newCount = $allFilesNow.Count - $baselineCount
            if ($newCount -ge $framesPerTakeResolved) {
                $selectedTakeFiles = $allFilesNow | Select-Object -Skip $baselineCount -First $framesPerTakeResolved
                break
            }

            Start-Sleep -Milliseconds $CapturePollIntervalMs
        }

        foreach ($file in $selectedTakeFiles) {
            Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $destJsonDir $file.Name)
        }

        $firstSelected = $selectedTakeFiles | Select-Object -First 1
        $lastSelected = $selectedTakeFiles | Select-Object -Last 1

        $parsedStartFrame = Get-FrameIndexFromJsonName -FileName $firstSelected.Name
        $parsedEndFrame = Get-FrameIndexFromJsonName -FileName $lastSelected.Name

        $fallbackStartFrame = $baselineCount
        $fallbackEndFrame = $baselineCount + $selectedTakeFiles.Count - 1

        $startFrameIndex = if ($null -ne $parsedStartFrame) { $parsedStartFrame } else { $fallbackStartFrame }
        $endFrameIndex = if ($null -ne $parsedEndFrame) { $parsedEndFrame } else { $fallbackEndFrame }

        $manifestRow = [pscustomobject]@{
            gesture           = $IdleGestureLabel
            person            = $person
            session           = $session
            take              = $nextTakeLabel
            start_frame_index = $startFrameIndex
            end_frame_index   = $endFrameIndex
            frames_copied     = $selectedTakeFiles.Count
            json_destination  = $destJsonDir
            review_video_path = $reviewVideoPath
        }

        if (Test-Path -LiteralPath $manifestCsvPath -PathType Leaf) {
            $manifestRow | Export-Csv -LiteralPath $manifestCsvPath -NoTypeInformation -Append
        }
        else {
            $manifestRow | Export-Csv -LiteralPath $manifestCsvPath -NoTypeInformation
        }

        $completedSampleCount++
        Write-Host ("Success: copied {0} frames to {1}" -f $selectedTakeFiles.Count, $destJsonDir) -ForegroundColor Green
    }
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
finally {
    Write-Host ""
    Write-Host "====================== Idle session summary ======================" -ForegroundColor Green
    Write-Host ("Idle dataset root : {0}" -f $idleSessionRoot) -ForegroundColor Green
    Write-Host ("Samples captured  : {0}" -f $completedSampleCount) -ForegroundColor Green
    Write-Host ("Manifest CSV      : {0}" -f $manifestCsvPath) -ForegroundColor Green
    if ($useReviewVideo) {
        Write-Host ("Review video      : {0}" -f $reviewVideoPath) -ForegroundColor Green
    }

    # Per requirement: do not force-kill by default.
    # Attempt one graceful close, then leave manual close if still running.
    if ($openPoseProcess -and -not $openPoseProcess.HasExited) {
        Write-Host ""
        Write-Host "Attempting one graceful OpenPose window close..." -ForegroundColor Yellow
        $null = $openPoseProcess.CloseMainWindow()
        Start-Sleep -Milliseconds 800

        if (-not $openPoseProcess.HasExited) {
            Write-Host "OpenPose is still running. Please close it manually when ready." -ForegroundColor Yellow
        }
        else {
            Write-Host "OpenPose closed gracefully." -ForegroundColor Green
        }
    }
}
