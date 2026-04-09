<#
.SYNOPSIS
Beginner-friendly timed OpenPose capture helper for Windows.

.DESCRIPTION
This script prompts for recording metadata (gesture/person/session/take),
creates dataset directories inside this repository, starts OpenPose, waits a
fixed duration, then automatically stops OpenPose.

It is designed for quick, repeatable takes (for example 3-second captures)
without changing any existing ML pipeline code.
#>

# ============================================================================
# EDITABLE DEFAULTS
# ----------------------------------------------------------------------------
# Update these paths for your machine. OpenPose must be installed outside this
# repository. This repo only stores the output data (JSON + optional video).
# ============================================================================
$OpenPoseRoot = "D:\\Programs\\OpenPose\\openpose"
$ProjectRoot = "D:\\Documentos\\Python Projects\\Avatar-Arcade-Game"

$DefaultPerson = "luis"
$DefaultSession = "s01"
$DefaultUseVideo = $true
$DefaultDurationSeconds = 3
$DefaultCountdownSeconds = 3

# ============================================================================
# Helper function: read input with an optional default value.
# If user presses ENTER with no text, we use the default.
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

Write-Host ""
Write-Host "=== OpenPose Timed Recording Helper ===" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Validate static paths early so users get immediate feedback.
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
# 1) Prompt user for recording metadata.
#    - gesture: required
#    - person: default to luis
#    - session: default to s01
#    - take: required
#    - save video: y/n default yes
#    - duration: default 3 seconds
# ============================================================================
$gesture = (Read-Value -Prompt "Gesture name (required)").Trim()
if ([string]::IsNullOrWhiteSpace($gesture)) {
    Write-Error "Gesture cannot be empty."
    exit 1
}

$person = (Read-Value -Prompt "Person" -Default $DefaultPerson).Trim()
if ([string]::IsNullOrWhiteSpace($person)) {
    $person = $DefaultPerson
}

$session = (Read-Value -Prompt "Session" -Default $DefaultSession).Trim()
if ([string]::IsNullOrWhiteSpace($session)) {
    $session = $DefaultSession
}

$take = (Read-Value -Prompt "Take label (required, e.g. take_001)").Trim()
if ([string]::IsNullOrWhiteSpace($take)) {
    Write-Error "Take cannot be empty."
    exit 1
}

$defaultVideoText = if ($DefaultUseVideo) { "y" } else { "n" }
$videoInput = (Read-Value -Prompt "Save video? (y/n)" -Default $defaultVideoText).Trim().ToLowerInvariant()
$useVideo = switch ($videoInput) {
    "" { $DefaultUseVideo }
    "y" { $true }
    "yes" { $true }
    "n" { $false }
    "no" { $false }
    default {
        Write-Warning "Unrecognized input '$videoInput'. Using default '$defaultVideoText'."
        $DefaultUseVideo
    }
}

$durationInput = (Read-Value -Prompt "Recording duration in seconds" -Default "$DefaultDurationSeconds").Trim()
[double]$durationSeconds = 0
if (-not [double]::TryParse($durationInput, [ref]$durationSeconds) -or $durationSeconds -le 0) {
    Write-Error "Duration must be a positive number."
    exit 1
}

# ============================================================================
# 2) Build output paths inside this repository.
# ============================================================================
$jsonDir = Join-Path $ProjectRoot ("data\\raw\\openpose_json\\{0}\\{1}\\{2}\\{3}" -f $gesture, $person, $session, $take)
$videoDir = Join-Path $ProjectRoot ("data\\raw\\rgb_video\\{0}\\{1}\\{2}" -f $gesture, $person, $session)
$videoPath = Join-Path $videoDir ("{0}.avi" -f $take)

# ============================================================================
# 3) Create required directories if they do not exist.
# ============================================================================
New-Item -ItemType Directory -Path $jsonDir -Force | Out-Null
if ($useVideo) {
    New-Item -ItemType Directory -Path $videoDir -Force | Out-Null
}

# ============================================================================
# 4) Print clear summary before recording starts.
# ============================================================================
Write-Host ""
Write-Host "=== Capture Summary ===" -ForegroundColor Yellow
Write-Host ("Gesture   : {0}" -f $gesture)
Write-Host ("Person    : {0}" -f $person)
Write-Host ("Session   : {0}" -f $session)
Write-Host ("Take      : {0}" -f $take)
Write-Host ("Duration  : {0} seconds" -f $durationSeconds)
Write-Host ("JSON Dir  : {0}" -f $jsonDir)
if ($useVideo) {
    Write-Host ("Video Path: {0}" -f $videoPath)
}
else {
    Write-Host "Video Path: (disabled)"
}

# ============================================================================
# 5) Countdown so the performer has time to get ready.
# ============================================================================
Write-Host ""
Write-Host "Get ready..." -ForegroundColor Cyan
for ($i = $DefaultCountdownSeconds; $i -ge 1; $i--) {
    Write-Host ("{0}..." -f $i)
    Start-Sleep -Seconds 1
}
Write-Host "GO" -ForegroundColor Green

# ============================================================================
# 6) Build OpenPose arguments.
#    Required flags:
#      --number_people_max 1
#      --tracking 1
#      --write_json <JSON_DIR>
#    Optional:
#      --write_video <VIDEO_PATH>
# ============================================================================
$openPoseArgs = @(
    "--number_people_max", "1",
    "--tracking", "1",
    "--write_json", $jsonDir
)

if ($useVideo) {
    $openPoseArgs += @("--write_video", $videoPath)
}

# ============================================================================
# 7) Start OpenPose as a child process, wait the requested duration, then stop.
# ============================================================================
$openPoseProcess = $null
try {
    $openPoseProcess = Start-Process -FilePath $OpenPoseExe -ArgumentList $openPoseArgs -WorkingDirectory $OpenPoseRoot -PassThru
    Write-Host ""
    Write-Host "OpenPose started (PID: $($openPoseProcess.Id)). Recording..." -ForegroundColor Green

    Start-Sleep -Milliseconds ([int]($durationSeconds * 1000))

    # Try graceful stop first.
    if (-not $openPoseProcess.HasExited) {
        Write-Host "Stopping OpenPose (graceful attempt)..." -ForegroundColor Yellow
        $null = $openPoseProcess.CloseMainWindow()
        Start-Sleep -Milliseconds 800
    }

    # If still alive, force stop.
    if (-not $openPoseProcess.HasExited) {
        Write-Host "OpenPose still running. Forcing stop..." -ForegroundColor Yellow
        Stop-Process -Id $openPoseProcess.Id -Force
    }

    # Brief wait so files flush to disk.
    Start-Sleep -Milliseconds 500

    Write-Host ""
    Write-Host "Capture complete." -ForegroundColor Green
    Write-Host ("JSON saved to : {0}" -f $jsonDir)
    if ($useVideo) {
        Write-Host ("Video saved to: {0}" -f $videoPath)
    }
    else {
        Write-Host "Video saving was disabled for this take."
    }
}
catch {
    Write-Error "Failed during capture: $($_.Exception.Message)"
    if ($openPoseProcess -and -not $openPoseProcess.HasExited) {
        Stop-Process -Id $openPoseProcess.Id -Force -ErrorAction SilentlyContinue
    }
    exit 1
}
