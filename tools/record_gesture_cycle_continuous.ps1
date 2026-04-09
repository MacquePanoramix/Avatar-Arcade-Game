<#
.SYNOPSIS
Beginner-friendly OpenPose helper for continuous 8-gesture cycle capture.

.DESCRIPTION
This script launches OpenPose ONE TIME and keeps it running while you record
an entire guided gesture cycle. OpenPose writes JSON files continuously into
a temporary "live buffer" folder.

For each gesture, the script captures one fixed-length take by copying exactly
N newly created JSON files from the live buffer into the repository dataset
folder for that gesture.

Why this helper exists:
- avoids restarting OpenPose between gestures
- keeps capture flow stable across the full cycle
- uses JSON-only output for this first version
#>

# ============================================================================
# EDITABLE DEFAULTS
# ----------------------------------------------------------------------------
# Update these paths for your machine.
# OpenPose is expected outside this repository.
# This repository stores copied JSON takes.
# ============================================================================
$OpenPoseRoot = "D:\Programs\OpenPose\openpose"
$ProjectRoot = "D:\Documentos\Python Projects\Avatar-Arcade-Game"

$DefaultPerson = "luis"
$DefaultSession = "s01"
$FramesPerTake = 90
$CapturePollIntervalMs = 200
$StartupTimeoutSeconds = 20
$LiveBufferRoot = Join-Path $ProjectRoot "data\\raw\\live_buffer"
$LiveSessionFolderName = "openpose_session"

# Fixed gesture cycle order.
$GestureCycle = @(
    "attack_earth",
    "attack_fire",
    "attack_water",
    "attack_air",
    "defense_earth",
    "defense_fire",
    "defense_water",
    "defense_air"
)

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
# Helper: determine one take label across the full cycle.
# Looks in: data/raw/openpose_json/<gesture>/<person>/<session>/take_###
# ============================================================================
function Get-NextTakeLabel {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRootPath,

        [Parameter(Mandatory = $true)]
        [string[]]$Gestures,

        [Parameter(Mandatory = $true)]
        [string]$Person,

        [Parameter(Mandatory = $true)]
        [string]$Session
    )

    $takeNumbers = New-Object System.Collections.Generic.List[int]

    foreach ($gestureName in $Gestures) {
        $sessionRoot = Join-Path $ProjectRootPath ("data\\raw\\openpose_json\\{0}\\{1}\\{2}" -f $gestureName, $Person, $Session)
        if (-not (Test-Path -LiteralPath $sessionRoot -PathType Container)) {
            continue
        }

        $takeDirs = Get-ChildItem -LiteralPath $sessionRoot -Directory -ErrorAction SilentlyContinue
        foreach ($takeDir in $takeDirs) {
            if ($takeDir.Name -match '^take_(\d+)$') {
                $takeNumbers.Add([int]$matches[1])
            }
        }
    }

    # If no takes exist yet, we always start at take_001.
    if ($takeNumbers.Count -eq 0) {
        return "take_001"
    }

    # Safer beginner-friendly formatting:
    # 1) Resolve max take as an integer.
    # 2) Add one.
    # 3) Build label using string concatenation + ToString("000").
    # This avoids edge cases where format placeholders can fail.
    [int]$maxTake = ($takeNumbers | Measure-Object -Maximum).Maximum
    [int]$nextTakeNumber = $maxTake + 1
    return "take_" + $nextTakeNumber.ToString("000")
}

# ============================================================================
# Helper: wait for live buffer JSON to start arriving.
# Also fails early if OpenPose exits before producing files.
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
    [bool]$firstJsonDetected = $false

    while ($stopwatch.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if ($OpenPoseProcess.HasExited) {
            throw "OpenPose exited before capture became live."
        }

        # Debug snapshot for beginners:
        # - exact live folder path
        # - whether first JSON has appeared yet
        # - current live JSON count
        $allJsonFiles = Get-ChildItem -LiteralPath $LiveRunFolder -Filter "*.json" -File -ErrorAction SilentlyContinue
        [int]$liveJsonCount = $allJsonFiles.Count
        $firstJson = $allJsonFiles | Select-Object -First 1
        if ($firstJson) {
            $firstJsonDetected = $true
            Write-Host ("[DEBUG] Live run folder      : {0}" -f $LiveRunFolder) -ForegroundColor DarkGray
            Write-Host ("[DEBUG] First JSON detected  : {0}" -f $firstJsonDetected) -ForegroundColor DarkGray
            Write-Host ("[DEBUG] Current JSON count   : {0}" -f $liveJsonCount) -ForegroundColor DarkGray
            return
        }

        Write-Host ("[DEBUG] Live run folder      : {0}" -f $LiveRunFolder) -ForegroundColor DarkGray
        Write-Host ("[DEBUG] First JSON detected  : {0}" -f $firstJsonDetected) -ForegroundColor DarkGray
        Write-Host ("[DEBUG] Current JSON count   : {0}" -f $liveJsonCount) -ForegroundColor DarkGray

        Start-Sleep -Milliseconds $PollIntervalMs
    }

    $filesAfterTimeout = Get-ChildItem -LiteralPath $LiveRunFolder -Filter "*.json" -File -ErrorAction SilentlyContinue
    [bool]$liveRunFolderStayedEmpty = ($filesAfterTimeout.Count -eq 0)
    throw ("Startup timed out after {0} seconds. Live run folder stayed empty: {1}. Live run folder path: {2}. No gesture capture has started yet." -f $TimeoutSeconds, $liveRunFolderStayedEmpty, $LiveRunFolder)
}

Write-Host ""
Write-Host "=== OpenPose Continuous Gesture Cycle Helper ===" -ForegroundColor Cyan
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
# 2) Prompt once for person/session/frames-per-take.
# ============================================================================
$person = (Read-Value -Prompt "Person" -Default $DefaultPerson).Trim()
if ([string]::IsNullOrWhiteSpace($person)) {
    $person = $DefaultPerson
}

$session = (Read-Value -Prompt "Session" -Default $DefaultSession).Trim()
if ([string]::IsNullOrWhiteSpace($session)) {
    $session = $DefaultSession
}

$framesInput = (Read-Value -Prompt "Frames per take" -Default "$FramesPerTake").Trim()
[int]$framesPerTakeResolved = 0
if (-not [int]::TryParse($framesInput, [ref]$framesPerTakeResolved) -or $framesPerTakeResolved -le 0) {
    Write-Error "Frames per take must be a positive integer."
    exit 1
}

# ============================================================================
# 3) Resolve next take label across full cycle.
# ============================================================================
$take = Get-NextTakeLabel -ProjectRootPath $ProjectRoot -Gestures $GestureCycle -Person $person -Session $session

Write-Host ""
Write-Host ("This cycle will use take label: {0}" -f $take) -ForegroundColor Yellow
Write-Host ("Frames per take: {0}" -f $framesPerTakeResolved) -ForegroundColor Yellow
Write-Host ""

# ============================================================================
# 4) Prepare live buffer folder and unique run folder.
# ============================================================================
$liveSessionFolder = Join-Path $LiveBufferRoot $LiveSessionFolderName
New-Item -ItemType Directory -Path $liveSessionFolder -Force | Out-Null

$runTimestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$liveRunFolder = Join-Path $liveSessionFolder ("session_{0}" -f $runTimestamp)
New-Item -ItemType Directory -Path $liveRunFolder -Force | Out-Null

$preexistingJson = Get-ChildItem -LiteralPath $liveRunFolder -Filter "*.json" -File -ErrorAction SilentlyContinue | Select-Object -First 1
if ($preexistingJson) {
    Write-Error "Live run folder already contains JSON files: $liveRunFolder"
    exit 1
}

# ============================================================================
# 5) Launch OpenPose ONCE with JSON output pointed to live run folder.
# ============================================================================
$openPoseArgs = "--number_people_max 1 --tracking 1 --write_json `"$liveRunFolder`""
$openPoseProcess = $null
$successfulGestures = New-Object System.Collections.Generic.List[string]

try {
    Write-Host "Launching OpenPose once for continuous capture..." -ForegroundColor Cyan
    Write-Host ("OpenPose arguments: {0}" -f $openPoseArgs) -ForegroundColor DarkGray

    $openPoseProcess = Start-Process -FilePath $OpenPoseExe -ArgumentList $openPoseArgs -WorkingDirectory $OpenPoseRoot -PassThru
    Write-Host ("OpenPose started (PID: {0})." -f $openPoseProcess.Id) -ForegroundColor Green

    # Wait until OpenPose is truly live (first JSON appears), with timeout.
    Write-Host ("Waiting for live JSON output (timeout: {0}s)..." -f $StartupTimeoutSeconds) -ForegroundColor Yellow
    Wait-ForLiveCaptureStart -OpenPoseProcess $openPoseProcess -LiveRunFolder $liveRunFolder -PollIntervalMs $CapturePollIntervalMs -TimeoutSeconds $StartupTimeoutSeconds
    Write-Host "Capture is live." -ForegroundColor Green
    Write-Host ("The first gesture will be: {0}" -f $GestureCycle[0]) -ForegroundColor Green
    Write-Host "Press ENTER when ready to start that gesture." -ForegroundColor Green
    $null = Read-Host

    # ========================================================================
    # 6) Guided cycle loop.
    # For each gesture:
    # - wait for user ENTER
    # - count current files
    # - wait for exactly N new files
    # - copy those files into dataset take folder
    # ========================================================================
    for ($index = 0; $index -lt $GestureCycle.Count; $index++) {
        $gesture = $GestureCycle[$index]
        $position = $index + 1

        $destJsonDir = Join-Path $ProjectRoot ("data\\raw\\openpose_json\\{0}\\{1}\\{2}\\{3}" -f $gesture, $person, $session, $take)

        Write-Host ""
        Write-Host "============================================================" -ForegroundColor DarkCyan
        Write-Host ("Gesture      : {0}" -f $gesture) -ForegroundColor Cyan
        Write-Host ("Cycle step   : {0}/{1}" -f $position, $GestureCycle.Count) -ForegroundColor Cyan
        if ($position -lt $GestureCycle.Count) {
            Write-Host ("Next gesture : {0}" -f $GestureCycle[$index + 1]) -ForegroundColor Cyan
        }
        else {
            Write-Host "Next gesture : (this is the last gesture)" -ForegroundColor Cyan
        }
        Write-Host ("Take label   : {0}" -f $take) -ForegroundColor Cyan

        $continueInput = (Read-Host "Press ENTER to start capturing this gesture, or type q to quit.").Trim().ToLowerInvariant()
        if ($continueInput -eq "q") {
            Write-Host "Cycle stopped by user request." -ForegroundColor Yellow
            break
        }

        # Refuse to overwrite existing JSON destination.
        New-Item -ItemType Directory -Path $destJsonDir -Force | Out-Null
        $existingDestJson = Get-ChildItem -LiteralPath $destJsonDir -Filter "*.json" -File -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($existingDestJson) {
            throw "Destination already contains JSON files: $destJsonDir"
        }

        # Snapshot current live buffer file count before this take starts.
        $baselineFiles = Get-ChildItem -LiteralPath $liveRunFolder -Filter "*.json" -File -ErrorAction SilentlyContinue |
            Sort-Object Name
        $baselineCount = $baselineFiles.Count

        Write-Host ("Capturing {0} new frames for '{1}'..." -f $framesPerTakeResolved, $gesture) -ForegroundColor Yellow

        # Wait for exactly N newly created files after baseline.
        $selectedTakeFiles = $null
        while ($true) {
            if ($openPoseProcess.HasExited) {
                throw "OpenPose exited unexpectedly during gesture '$gesture'."
            }

            $allFilesNow = Get-ChildItem -LiteralPath $liveRunFolder -Filter "*.json" -File -ErrorAction SilentlyContinue |
                Sort-Object Name

            $newCount = $allFilesNow.Count - $baselineCount

            if ($newCount -ge $framesPerTakeResolved) {
                # Deterministic selection:
                # all files are sorted by filename, and we select the first N files
                # after the baseline point.
                $selectedTakeFiles = $allFilesNow | Select-Object -Skip $baselineCount -First $framesPerTakeResolved
                break
            }

            Start-Sleep -Milliseconds $CapturePollIntervalMs
        }

        # Copy (not move) exactly N files to gesture destination.
        foreach ($file in $selectedTakeFiles) {
            Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $destJsonDir $file.Name)
        }

        $successfulGestures.Add($gesture)
        Write-Host ("Success: copied {0} frames to {1}" -f $selectedTakeFiles.Count, $destJsonDir) -ForegroundColor Green
    }
}
catch {
    Write-Error $_.Exception.Message
    exit 1
}
finally {
    Write-Host ""
    Write-Host "====================== Cycle summary =======================" -ForegroundColor Green
    Write-Host ("Take label used: {0}" -f $take) -ForegroundColor Green
    Write-Host "Successful gestures:" -ForegroundColor Green

    if ($successfulGestures.Count -eq 0) {
        Write-Host "- (none)"
    }
    else {
        foreach ($recordedGesture in $successfulGestures) {
            Write-Host ("- {0}" -f $recordedGesture)
        }
    }

    # Per requirement: do not force-kill by default.
    # We attempt one graceful close, then ask user to close manually if needed.
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
