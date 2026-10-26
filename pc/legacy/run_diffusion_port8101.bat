@echo off
REM Portable manual start: lazy server on 0.0.0.0:8101 (override with
REM DIFFUSION_PORT / DIFFUSION_HOST env vars). No admin needed.
set SERVER_DIR=%~dp0
set SERVER_DIR=%SERVER_DIR:~0,-1%
set VENV_PYTHON=%SERVER_DIR%\.venv\Scripts\python.exe
if not exist "%VENV_PYTHON%" (
  echo Missing venv python: %VENV_PYTHON%
  echo Create it: py -3.12 -m venv "%SERVER_DIR%\.venv" ^&^& install requirements.txt
  exit /b 1
)

if "%DIFFUSION_PORT%"=="" set DIFFUSION_PORT=8101
if "%DIFFUSION_HOST%"=="" set DIFFUSION_HOST=0.0.0.0

set HF_HOME=%SERVER_DIR%\.hf_cache
set HF_HUB_CACHE=%SERVER_DIR%\.hf_cache\hub
set HUGGINGFACE_HUB_CACHE=%SERVER_DIR%\.hf_cache\hub
set HF_HUB_DISABLE_TELEMETRY=1
set HF_HUB_VERBOSITY=warning
set TRANSFORMERS_VERBOSITY=warning
set TRANSFORMERS_NO_TF=1
set TRANSFORMERS_NO_FLAX=1
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /d %SERVER_DIR%

set ALLOWED=
for /f %%i in ('tailscale ip -4 2^>nul') do set ALLOWED=--allowed-host %%i

"%VENV_PYTHON%" diffusion_server.py --lazy --models-json "%SERVER_DIR%\models.json" --port %DIFFUSION_PORT% --host %DIFFUSION_HOST% %ALLOWED% --disable-safety-checker
