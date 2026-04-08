@echo off
setlocal EnableExtensions

REM =============================================================
REM  OpenPose Recording Helper (Windows)
REM  This script launches OpenPose and saves JSON/video into this repo.
REM =============================================================

REM =============================================================
REM  EDIT THESE VALUES BEFORE RECORDING
REM =============================================================
REM Example:
REM set "OPENPOSE_ROOT=D:\Programs\OpenPose\openpose"
set "OPENPOSE_ROOT=D:\Programs\OpenPose\openpose"

REM Example:
REM set "PROJECT_ROOT=D:\Documentos\Python Projects\Avatar-Arcade-Game"
set "PROJECT_ROOT=D:\Documentos\Python Projects\Avatar-Arcade-Game"

REM Label for the gesture class (required)
set "GESTURE="

REM Person/performer identifier
set "PERSON=luis"

REM Session identifier
set "SESSION=s01"

REM Take identifier (required), e.g. take_001
set "TAKE="

REM 1 = save video (.avi), 0 = JSON only
set "USE_VIDEO=1"

REM =============================================================
REM  SAFETY CHECKS
REM =============================================================
if "%GESTURE%"=="" (
    echo [ERROR] GESTURE is empty. Please set GESTURE in the EDIT section.
    exit /b 1
)

if "%TAKE%"=="" (
    echo [ERROR] TAKE is empty. Please set TAKE in the EDIT section.
    exit /b 1
)

if not exist "%OPENPOSE_ROOT%" (
    echo [ERROR] OPENPOSE_ROOT does not exist: "%OPENPOSE_ROOT%"
    exit /b 1
)

if not exist "%PROJECT_ROOT%" (
    echo [ERROR] PROJECT_ROOT does not exist: "%PROJECT_ROOT%"
    exit /b 1
)

if not exist "%OPENPOSE_ROOT%\bin\OpenPoseDemo.exe" (
    echo [ERROR] OpenPose executable not found:
    echo         "%OPENPOSE_ROOT%\bin\OpenPoseDemo.exe"
    exit /b 1
)

if not "%USE_VIDEO%"=="1" if not "%USE_VIDEO%"=="0" (
    echo [ERROR] USE_VIDEO must be 1 or 0. Current value: "%USE_VIDEO%"
    exit /b 1
)

REM =============================================================
REM  BUILD OUTPUT PATHS (do not edit)
REM =============================================================
set "JSON_DIR=%PROJECT_ROOT%\data\raw\openpose_json\%GESTURE%\%PERSON%\%SESSION%\%TAKE%\"
set "VIDEO_PATH=%PROJECT_ROOT%\data\raw\rgb_video\%GESTURE%\%PERSON%\%SESSION%\%TAKE%.avi"

if not exist "%JSON_DIR%" (
    mkdir "%JSON_DIR%"
)

REM Ensure parent folder for video exists if video is enabled
if "%USE_VIDEO%"=="1" (
    if not exist "%PROJECT_ROOT%\data\raw\rgb_video\%GESTURE%\%PERSON%\%SESSION%\" (
        mkdir "%PROJECT_ROOT%\data\raw\rgb_video\%GESTURE%\%PERSON%\%SESSION%\"
    )
)

REM =============================================================
REM  SHOW RESOLVED SETTINGS
REM =============================================================
echo.
echo ================== Recording Settings ==================
echo OPENPOSE_ROOT : %OPENPOSE_ROOT%
echo PROJECT_ROOT  : %PROJECT_ROOT%
echo GESTURE       : %GESTURE%
echo PERSON        : %PERSON%
echo SESSION       : %SESSION%
echo TAKE          : %TAKE%
echo USE_VIDEO     : %USE_VIDEO%
echo JSON_DIR      : %JSON_DIR%
if "%USE_VIDEO%"=="1" (
    echo VIDEO_PATH    : %VIDEO_PATH%
) else (
    echo VIDEO_PATH    : ^(disabled^)
)
echo ========================================================
echo.

REM =============================================================
REM  RUN OPENPOSE
REM =============================================================
pushd "%OPENPOSE_ROOT%"

if "%USE_VIDEO%"=="1" (
    .\bin\OpenPoseDemo.exe --number_people_max 1 --tracking 1 --write_json "%JSON_DIR%" --write_video "%VIDEO_PATH%"
) else (
    .\bin\OpenPoseDemo.exe --number_people_max 1 --tracking 1 --write_json "%JSON_DIR%"
)

set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] OpenPose finished with exit code %EXIT_CODE%.
    exit /b %EXIT_CODE%
)

echo.
echo [DONE] Recording complete.
echo [SAVED] JSON files: "%JSON_DIR%"
if "%USE_VIDEO%"=="1" echo [SAVED] Video file: "%VIDEO_PATH%"

echo.
echo Tip: For each take, perform neutral pose ^> gesture ^> return to neutral.

endlocal
exit /b 0
