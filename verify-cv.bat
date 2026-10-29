@echo off
cd /d "%~dp0backend"
"..\.venv\Scripts\python.exe" verify_cv.py
echo.
pause
