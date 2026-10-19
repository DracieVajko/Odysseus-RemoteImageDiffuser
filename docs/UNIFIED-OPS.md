# UNIFIED-OPS — prevádzka unified servera (bez pushu, lokálne)

## Štart / stop (Windows, .venv)

```powershell
.\Start-Unified.ps1            # :8101 lazy, whisper fallback -> :4444
.\Start-Unified.ps1 -Remote   # + bind 0.0.0.0 (Tailscale/LAN)
.\Start-Unified.ps1 -Shim     # + legacy :4444 -> :8101 pre starých klientov
.\Stop-Unified.ps1
```

Starý `Start-Diffusion.ps1` / `Stop-Diffusion.ps1` ostáva nedotknutý (fix
branch). Nové PID súbory: `run/unified.pid`, `run/unified-shim.pid`.
Env: `UNIFIED_WHISPER_URL` (default `http://127.0.0.1:4444`),
`UNIFIED_NO_ENGINE=1` = bez diffusion registrácie (vždy placeholder).

## CLI

```powershell
.venv\Scripts\python.exe generate_image.py doctor
.venv\Scripts\python.exe generate_image.py txt2img --model DreamShaper_8 --prompt "a cabin" --output out.png
.venv\Scripts\python.exe generate_image.py upscale --input out.png --scale 2 --output out2x.png
.venv\Scripts\python.exe generate_image.py transcribe --input audio.mp3 --model small --language sk
```

## Modely

- Obrazy: `models.json` (4× SD, ~19 GB) + `download_models.py --preset odysseus-default|sd15|sdxl|sdxl-turbo`.
- Upscale: `models/RealESRGAN_x4v3/RealESRGAN_x4plus.pth` (67 MB, BSD-3)
  cez `download_models.py --preset upscale`. Bez váhy = Pillow fallback.
- Audio: WhisperApi služba na :4444 (vlastný faster-whisper env) alebo
  `pip install faster-whisper` do `.venv` pre lokálny engine.
- VLM: `download_models.py --preset vlm` (SmolVLM-500M, ~1 GB).

### Install / uninstall z WebUI a CLI

WebUI → tab **Models**: zoznam modelov, stav installed/not installed,
tlačidlá **Install** / **Uninstall**, progress a „Install all missing".

CLI:
```powershell
.venv\Scripts\python.exe generate_image.py models            # katalóg + stav
.venv\Scripts\python.exe generate_image.py install SDXL_Turbo
.venv\Scripts\python.exe generate_image.py uninstall SDXL_Turbo
.venv\Scripts\python.exe generate_image.py status            # server + modely
```

API:
- `GET  /v1/models/catalog`
- `POST /v1/models/install    {"id": "..."}`
- `POST /v1/models/uninstall  {"id": "..."}`
- `GET  /v1/settings`

Uninstall má path guard: nikdy nezmaže nič mimo `models/`.

## FFmpeg tools (WebUI tab Tools + CLI)

```powershell
python generate_image.py presets
python generate_image.py video --input in.mp4 --preset mp4-small --resize 720p --output out.mp4
python generate_image.py audio --input in.wav --preset mp3-128 --output out.mp3
python generate_image.py extract --input video.mp4 --format wav --output a.wav
python generate_image.py trim --input in.mp4 --start 1 --end 5 --output cut.mp4
python generate_image.py compress --input in.mp4 --size-mb 10 --output small.mp4
python generate_image.py info --input in.mp4
```

API: `GET /v1/ffmpeg/presets`, `POST /v1/ffmpeg/{convert-video|convert-audio|
extract-audio|trim|compress|info}` (multipart).

## Prepnutie Windows služby na unified

`OdysseusDiffusion` služba dnes beží starý `diffusion_server.py`. Prepnutie
(mení bežiacu službu, spusti vedome):

```powershell
.\Switch-Service-ToUnified.ps1 -WhatIf   # ukáže, čo by zmenil
.\Switch-Service-ToUnified.ps1           # prepne na unified_server
.\Switch-Service-ToUnified.ps1 -Revert   # vráti starý diffusion_server
```

## Odysseus wiring

- Image base URL: `http://<host>:8101`, audio base URL: `http://<host>:8101`
  (cesty `/v1/audio/...` identické s bývalou :4444).
- Zmluva: `docs/UNIFIED-CONTRACT.md` (frozen). Starý kontrakt platí ďalej:
  `docs/ODYSSEUS-CONTRACT.md`.

## Portable balíček (Win/Mac/Linux)

V1 = repo + `.venv` + `models/` + `Start-Unified.ps1` (Win) /
`uvicorn unified_server:app` (.venv python) všade. Mac: `--cpu` cesta
automaticky (torch MPS/CPU, SD pomalé ale funkčné); Linux: rovnaký venv
postup ako Win. Plný inštalátor (PyInstaller/Briefcase) až po stabilizácii
V1 — venv má ~8 GB, modely ~19 GB, nič z toho necommitovať.

## Android

Pozri `android/README.md` + `android/contract-compat.md` a `docs/ANDROID.md`.
`./gradlew :app:assembleDebug` až na stroji s Android SDK.

Android appka má vlastné UI (status karta, Local/Auto/Remote, modely
s Install/Uninstall a reálnym progress barom, activity log) a navyše
**Open Web UI** — servíruje rovnaké `webui_unified.html` z APK assets,
takže UI je identické s PC. WebUI sa do APK syncuje pri každom builde
(Gradle task `syncWebUi`).

Na mobile funguje reálne: `/health`, `/v1/models`, `/v1/audio/*` (whisper JNI),
`/v1/models/catalog|install|uninstall`, `/v1/settings`, `/webui`.
Generovanie obrázkov / upscale / caption vracajú 503 (mobilný engine nie je
hotový) — vtedy použi Remote mode na PC.

### Test cez adb

```powershell
powershell -ExecutionPolicy Bypass -File android/adb_test.ps1
```
Skript: forward `8199→8101`, health, modely, transcribe smoke, caption 503 check,
logcat.
