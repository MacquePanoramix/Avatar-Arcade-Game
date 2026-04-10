@echo off
setlocal EnableExtensions

REM =============================================================
REM  Idle pose recorder launcher (double-click friendly)
REM  - Runs the continuous idle PowerShell helper
REM  - Uses ExecutionPolicy Bypass for this process only
REM  - Starts from the repository root
REM =============================================================

set "REPO_ROOT=%~dp0.."
pushd "%REPO_ROOT%"

powershell -NoProfile -ExecutionPolicy Bypass -File ".\tools\record_idle_pose_continuous.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

popd
endlocal & exit /b %EXIT_CODE%
