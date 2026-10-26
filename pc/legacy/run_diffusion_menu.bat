@echo off
REM ============================================================
REM Odysseus Diffusion Server - Model Selection & Generation Menu
REM Works with the lazy-loading server running on port 8101
REM ============================================================

setlocal enabledelayedexpansion

set SERVER_DIR=D:\AI\DiffusionServer
set VENV_PYTHON=%SERVER_DIR%\.venv\Scripts\python.exe
set SERVER_URL=http://127.0.0.1:8101
set OUTPUT_DIR=%SERVER_DIR%\tests\output

REM ============================================================
REM MODEL CONFIGURATION
REM ============================================================
set MODEL_COUNT=4
set MODEL_1_ID=DreamShaper_8
set MODEL_1_NAME=DreamShaper 8 (General/Art/Realism, SD1.5)
set MODEL_1_BASE=768x768
set MODEL_1_MAX=1024x1024

set MODEL_2_ID=AbsoluteReality_1.8
set MODEL_2_NAME=Absolute Reality 1.8.1 (Photorealism, SD1.5)
set MODEL_2_BASE=768x768
set MODEL_2_MAX=1024x1024

set MODEL_3_ID=AnimagineXL_3.1
set MODEL_3_NAME=Animagine XL 3.1 (Anime, SDXL)
set MODEL_3_BASE=1024x1024
set MODEL_3_MAX=1920x1080

set MODEL_4_ID=JuggernautXL_v9
set MODEL_4_NAME=Juggernaut XL v9 (Photorealistic, SDXL)
set MODEL_4_BASE=1024x1024
set MODEL_4_MAX=1920x1080

REM ============================================================
REM NEGATIVE PROMPT (from config)
REM ============================================================
set NEGATIVE_PROMPT=blurry, low quality, low resolution, bad anatomy, deformed, distorted, extra fingers, missing fingers, bad hands, text, watermark, oversaturated, artifacts

:MAIN_MENU
cls
echo ============================================================
echo   ODYSSEUS DIFFUSION - MODEL SELECTION ^& GENERATION
echo ============================================================
echo.
echo Server: %SERVER_URL%
echo Output: %OUTPUT_DIR%
echo.
echo Available Models:
echo.
echo   1) %MODEL_1_NAME%
echo      Base: %MODEL_1_BASE% ^| Max: %MODEL_1_MAX%
echo.
echo   2) %MODEL_2_NAME%
echo      Base: %MODEL_2_BASE% ^| Max: %MODEL_2_MAX%
echo.
echo   3) %MODEL_3_NAME%
echo      Base: %MODEL_3_BASE% ^| Max: %MODEL_3_MAX%
echo.
echo   4) %MODEL_4_NAME%
echo      Base: %MODEL_4_BASE% ^| Max: %MODEL_4_MAX%
echo.
echo   Q) Quit
echo.
echo ============================================================

set /p CHOICE=Select model (1-4 or Q): 
echo.

if /i "!CHOICE!"=="Q" goto :EOF
if "!CHOICE!"=="1" set SELECTED_ID=%MODEL_1_ID% & set SELECTED_NAME=%MODEL_1_NAME% & set SELECTED_BASE=%MODEL_1_BASE% & set SELECTED_MAX=%MODEL_1_MAX% & goto PROMPT_MENU
if "!CHOICE!"=="2" set SELECTED_ID=%MODEL_2_ID% & set SELECTED_NAME=%MODEL_2_NAME% & set SELECTED_BASE=%MODEL_2_BASE% & set SELECTED_MAX=%MODEL_2_MAX% & goto PROMPT_MENU
if "!CHOICE!"=="3" set SELECTED_ID=%MODEL_3_ID% & set SELECTED_NAME=%MODEL_3_NAME% & set SELECTED_BASE=%MODEL_3_BASE% & set SELECTED_MAX=%MODEL_3_MAX% & goto PROMPT_MENU
if "!CHOICE!"=="4" set SELECTED_ID=%MODEL_4_ID% & set SELECTED_NAME=%MODEL_4_NAME% & set SELECTED_BASE=%MODEL_4_BASE% & set SELECTED_MAX=%MODEL_4_MAX% & goto PROMPT_MENU

echo Invalid choice. Press any key to continue...
pause ^>nul
goto MAIN_MENU

:PROMPT_MENU
cls
echo ============================================================
echo   SELECTED: %SELECTED_NAME% (%SELECTED_ID%)
echo   Base Resolution: %SELECTED_BASE% ^| Max: %SELECTED_MAX%
echo ============================================================
echo.
echo Enter your POSITIVE PROMPT (what you want to generate):
echo.
set /p POSITIVE_PROMPT=Prompt: 
echo.

if "!POSITIVE_PROMPT!"=="" (
    echo Prompt cannot be empty. Press any key to continue...
    pause >nul
    goto PROMPT_MENU
)

:SIZE_MENU
cls
echo ============================================================
echo Selected: %SELECTED_NAME%
echo Prompt: %POSITIVE_PROMPT%
echo Negative: %NEGATIVE_PROMPT%
echo ============================================================
echo.
echo Resolution Options:
echo   1) Base Resolution (%SELECTED_BASE%)
echo   2) Max Resolution (%SELECTED_MAX%)
echo   3) Custom Size (e.g. 512x512)
echo.
set /p SIZE_CHOICE=Select resolution (1-3): 
echo.

if "!SIZE_CHOICE!"=="1" set SIZE=%SELECTED_BASE%
if "!SIZE_CHOICE!"=="2" set SIZE=%SELECTED_MAX%
if "!SIZE_CHOICE!"=="3" (
    set /p SIZE=Enter custom size (WIDTHxHEIGHT): 
    if "!SIZE!"=="" (
        echo Invalid size. Press any key...
        pause ^>nul
        goto SIZE_MENU
    )
)

if "!SIZE!"=="" (
    echo Invalid choice. Press any key...
    pause ^>nul
    goto SIZE_MENU
)

:STEPS_MENU
echo.
echo Inference Steps (quality vs speed):
echo   1) Fast (15 steps)
echo   2) Balanced (25 steps) - Recommended
echo   3) High Quality (35 steps)
echo   4) Custom
echo.
set /p STEPS_CHOICE=Select steps (1-4): 
echo.

if "!STEPS_CHOICE!"=="1" set STEPS=15
if "!STEPS_CHOICE!"=="2" set STEPS=25
if "!STEPS_CHOICE!"=="3" set STEPS=35
if "!STEPS_CHOICE!"=="4" (
    set /p STEPS=Enter custom steps (10-100): 
    if "!STEPS!"=="" set STEPS=25
)

:FILENAME_MENU
echo.
echo Output Filename (without extension):
set /p FILENAME=Filename (empty = auto): 
echo.

if "!FILENAME!"=="" (
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set datetime=%%I
    set timestamp=!datetime:~0,4!!datetime:~4,2!!datetime:~6,2!-!datetime:~8,2!!datetime:~10,2!!datetime:~12,2!
    set FILENAME=%SELECTED_ID%_!timestamp!
)

:CONFIRM
cls
echo ============================================================
echo   GENERATION CONFIRMATION
echo ============================================================
echo Model:       %SELECTED_NAME% (%SELECTED_ID%)
echo Prompt:      %POSITIVE_PROMPT%
echo Negative:    %NEGATIVE_PROMPT%
echo Resolution:  %SIZE%
echo Steps:       %STEPS%
echo Filename:    %FILENAME%.png
echo Output:      %OUTPUT_DIR%
echo ============================================================
echo.
set /p CONFIRM=Generate? (Y/N): 
echo.

if /i "!CONFIRM!"=="Y" goto DO_GENERATE
if /i "!CONFIRM!"=="N" goto PROMPT_MENU
goto CONFIRM

:DO_GENERATE
cls
echo ============================================================
echo   GENERATING...
echo ============================================================
echo Model: %SELECTED_NAME% (%SELECTED_ID%)
echo Prompt: %POSITIVE_PROMPT%
echo Size: %SIZE%
echo Steps: %STEPS%
echo.
echo Requesting generation...
echo.

REM Call Python generation script
set PYTHON_SCRIPT=%SERVER_DIR%\generate_image.py
if not exist "%PYTHON_SCRIPT%" (
    echo Creating generation script...
    call :CREATE_GEN_SCRIPT
)

python "%PYTHON_SCRIPT%" ^
    --model "%SELECTED_ID%" ^
    --prompt "%POSITIVE_PROMPT%" ^
    --negative "%NEGATIVE_PROMPT%" ^
    --size %SIZE% ^
    --steps %STEPS% ^
    --output "%OUTPUT_DIR%\%FILENAME%.png" ^
    --server "%SERVER_URL%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Generation failed with error code %ERRORLEVEL%
    pause
    goto PROMPT_MENU
)

echo.
echo ============================================================
echo Generation complete! Saved to: %OUTPUT_DIR%\%FILENAME%.png
echo ============================================================
echo.
set /p CONT=Generate another? (Y/N): 
if /i "!CONT!"=="Y" goto PROMPT_MENU
goto MAIN_MENU

:CREATE_GEN_SCRIPT
(
echo import sys
echo import json
echo import base64
echo import requests
echo import argparse
echo import time
echo.
echo parser = argparse.ArgumentParser()
echo parser.add_argument('--model', required=True)
echo parser.add_argument('--prompt', required=True)
echo parser.add_argument('--negative', default='')
echo parser.add_argument('--size', default='512x512')
echo parser.add_argument('--steps', type=int, default=25)
echo parser.add_argument('--output', required=True)
echo parser.add_argument('--server', default='http://127.0.0.1:8101')
echo args = parser.parse_args()
echo.
echo width, height = map(int, args.size.split('x'))
echo.
echo payload = {
echo     'model': args.model,
echo     'prompt': args.prompt,
echo     'size': args.size,
echo     'n': 1,
echo     'quality': 'medium',
echo     'steps': args.steps
echo }
echo if args.negative:
echo     payload['negative_prompt'] = args.negative
echo.
echo url = f'{args.server}/v1/images/generations'
echo headers = {'Content-Type': 'application/json'}
echo.
echo print(f'Requesting generation: {args.model} @ {args.size} ({args.steps} steps)')
echo start = time.time()
echo response = requests.post(url, json=payload, headers=headers, timeout=600)
echo elapsed = time.time() - start
echo.
echo if response.status_code == 200:
echo     data = response.json()
echo     b64 = data['data'][0]['b64_json']
echo     import base64
echo     with open(args.output, 'wb') as f:
echo         f.write(base64.b64decode(b64))
echo     print(f'Generation complete in {{elapsed:.1f}}s -> {{args.output}}')
echo else:
echo     print(f'Error {{response.status_code}}: {{response.text}}')
echo     sys.exit(1)
) > "%SERVER_DIR%\generate_image.py"
goto :EOF