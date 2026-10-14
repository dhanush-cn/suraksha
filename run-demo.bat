@echo off
REM Double-click this to start the RockfallGuard demo on http://localhost:8005
REM Bypasses the PowerShell execution policy for this one process only.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-demo.ps1"
pause
