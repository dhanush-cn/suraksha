@echo off
REM Trains the pit-region CNN on the real LabelMe annotations.
REM Takes roughly 15-25 minutes on CPU. Keep the laptop plugged in.
cd /d "%~dp0"
".venv\Scripts\python.exe" ml\train_cnn.py
echo.
echo Finished. Press any key to close.
pause
