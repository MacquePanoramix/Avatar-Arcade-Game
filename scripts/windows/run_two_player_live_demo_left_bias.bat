@echo off
setlocal

set "REPO_DIR=D:\Documentos\Python Projects\Avatar-Arcade-Game"
set "OPENPOSE_DIR=D:\Programs\OpenPose\openpose"
set "MODEL_PATH=D:\Documentos\Python Projects\Avatar-Arcade-Game\models\checkpoints\best_mlp.keras"
set "OPENPOSE_MODEL_DIR=D:\Programs\OpenPose\openpose\models"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"

set "FOLDER=%REPO_DIR%\data\raw\live_buffer\openpose_session\live_test_%TS%"
set "JSON_OUT=%REPO_DIR%\logs\inference\latest_prediction_%TS%.json"
set "JSONL_OUT=%REPO_DIR%\logs\inference\prediction_stream_%TS%.jsonl"
set "CSV_OUT=%REPO_DIR%\logs\inference\live_test_%TS%.csv"

if not exist "%REPO_DIR%\data\raw\live_buffer\openpose_session" mkdir "%REPO_DIR%\data\raw\live_buffer\openpose_session"
if not exist "%REPO_DIR%\logs\inference" mkdir "%REPO_DIR%\logs\inference"
if not exist "%FOLDER%" mkdir "%FOLDER%"

echo ========================================================
echo Avatar Arcade Live Demo (Two Player Left-Bias Fallback)
echo Timestamp: %TS%
echo OpenPose JSON folder: %FOLDER%
echo Latest JSON output: %JSON_OUT%
echo JSONL stream output: %JSONL_OUT%
echo CSV log output: %CSV_OUT%
echo ========================================================

start "OpenPose Live (Two Player Left-Bias)" cmd /k "cd /d \"%OPENPOSE_DIR%\" ^&^& .\bin\OpenPoseDemo.exe --write_json \"%FOLDER%\" --model_folder \"%OPENPOSE_MODEL_DIR%\" --display 1 --render_pose 1"

start "Classifier Live (Two Player Left-Bias)" cmd /k "cd /d \"%REPO_DIR%\" ^&^& python -m src.inference.live_openpose_debug --json-dir \"%FOLDER%\" --model-path \"%MODEL_PATH%\" --tracking-mode two_player_left_right --side-split-x 380 --overlay-mode both --live-source-fps 12 --require-motion-for-nonidle --accept-threshold 0.55 --margin-threshold 0.02 --trigger-streak 1 --trigger-cooldown-frames 5 --release-idle-frames 1 --motion-on-min-consecutive 1 --active-span-min-frames 2 --output-latest-json \"%JSON_OUT%\" --output-jsonl \"%JSONL_OUT%\" --log-csv \"%CSV_OUT%\" --max-idle-polls 0"

endlocal
