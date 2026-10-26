@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0pc\scripts\Stop-Unified.ps1"
pause
endlocal
