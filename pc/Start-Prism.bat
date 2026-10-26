@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\Start-Prism.ps1" %*
if errorlevel 1 pause
endlocal
