# UNIFIED-OPS — operating the unified server (no push, local)

All paths below are relative to the repository root. The PC product lives in
`pc/`; run the server from there (or use the prefixed paths).

## Start / stop (Windows, .venv)

```powershell
.\pc\scripts\Start-Unified.ps1            # :8101 lazy, whisper fallback -> :4444
.\pc\scripts\Start-Unified.ps1 -Remote    # + bind 0.0.0.0 (Tailscale/LAN)
.\pc\scripts\Start-Unified.ps1 -Shim      # + legacy :4444 -> :8101 for old clients
.\pc\scripts\Stop-Unified.ps1
```

The scripts look for a venv in `pc/.venv` first, then the repo-root `.venv`.
The old `Start-Diffusion.ps1` / `Stop-Diffusion.ps1` lives in `pc/legacy/`
(kept for reference). New PID files: `pc/run/unified.pid`,
`pc/run/unified-shim.pid`.
Env: `UNIFIED_WHISPER_URL` (default `http://127.0.0.1:4444`),
`UNIFIED_NO_ENGINE=1` = without diffusion registration (always placeholder).

## CLI

```powershell
.venv\Scripts\python.exe pc\generate_image.py doctor
.venv\Scripts\python.exe pc\generate_image.py txt2img --model DreamShaper_8 --prompt "a cabin" --output out.png
.venv\Scripts\python.exe pc\generate_image.py upscale --input out.png --scale 2 --output out2x.png
.venv\Scripts\python.exe pc\generate_image.py transcribe --input audio.mp3 --model small --language sk
```

## Models

- Images: `models.json` (4× SD, ~19 GB) + `pc/download_models.py --preset odysseus-default|sd15|sdxl|sdxl-turbo`.
- Upscale: `models/RealESRGAN_x4v3/RealESRGAN_x4plus.pth` (67 MB, BSD-3)
  via `pc/download_models.py --preset upscale`. Without weights = Pillow fallback.
- Audio: WhisperApi service on :4444 (its own faster-whisper env) or
  `pip install faster-whisper` into `.venv` for the local engine.
- VLM: `pc/download_models.py --preset vlm` (SmolVLM-500M, ~1 GB).

### GPU (NVIDIA CUDA / AMD DirectML)

```powershell
pip install onnxruntime-gpu nvidia-cudnn-cu13   # NVIDIA (CUDA 13 + cuDNN 9)
pip install torch --index-url https://download.pytorch.org/whl/cu130  # VLM + ESRGAN (CUDA)
pip install transformers                        # SmolVLM caption/chat
# or, in a separate env, for AMD:
pip install onnxruntime-directml
```

- `PRISM_ONNX_PROVIDER=auto|cuda|dml|cpu` (default `auto`).
- `auto` prefers CUDA; DirectML is skipped for the INT8 QDQ LCM model because
  DML cannot fuse that graph.
- `/health` shows `device` and `onnx_provider`.
- The 6 GB RTX 3060 runs at ~5.9 GB VRAM; use `PRISM_ONNX_PROVIDER=cpu` if
  other GPU apps need memory. Switching to upscale/VLM releases the LCM
  sessions first.
- AMD: the current INT8 QDQ asset runs on CPU only; DirectML needs a
  non-QDQ (FP16/FP32) ONNX model.

### Upscale

- `pc/download_models.py --preset upscale` fetches `RealESRGAN_x4plus.pth`.
- Convert once for torch-free inference:
  `python pc/tools/convert_esrgan_onnx.py` → `RealESRGAN_x4plus.onnx` (67 MB, single file).
- `PRISM_UPSCALE_ENGINE=auto|torch|onnx|pillow` (default `auto`:
  torch → ONNX → Pillow). Measured 512→1024 on CUDA: ~6 s.

### VLM

- `pc/download_models.py --preset vlm` fetches the SmolVLM-500M snapshot (~1 GB).
- Requires `torch` + `transformers`; without them the endpoint returns
  503 will-not-load (never fake output).

### Install / uninstall from WebUI and CLI

WebUI → **Models** tab: model list, installed/not installed status,
**Install** / **Uninstall** buttons, progress and "Install all missing".

CLI:
```powershell
.venv\Scripts\python.exe generate_image.py models            # catalog + status
.venv\Scripts\python.exe generate_image.py install SDXL_Turbo
.venv\Scripts\python.exe generate_image.py uninstall SDXL_Turbo
.venv\Scripts\python.exe generate_image.py status            # server + models
```

API:
- `GET  /v1/models/catalog`
- `POST /v1/models/install    {"id": "..."}`
- `POST /v1/models/uninstall  {"id": "..."}`
- `GET  /v1/settings`

Uninstall has a path guard: it never deletes anything outside `models/`.

### Model finder + custom models (PC)

```powershell
python generate_image.py search dreamshaper --limit 10   # HuggingFace search
python generate_image.py add Lykon/DreamShaper            # downloads into models/DreamShaper
python generate_image.py custom                          # list of custom directories
python generate_image.py delete DreamShaper              # deletes a custom model
```

API: `GET /v1/models/find?q=`, `POST /v1/models/add {repo_id|url,name}`,
`GET /v1/models/custom`, `POST /v1/models/delete {name}`.

Rules:
- `add` with `repo_id` (diffusers snapshot) is PC-only; Android supports
  single-file URL (GGUF/ONNX/safetensors) via DownloadManager.
- `delete` refuses catalog models (those use `uninstall`) and paths
  outside `models/`.

## Lazy loading and idle unload

- Start: **no model in RAM/VRAM** (`/health` → `current_model: ""`,
  `idle_seconds: null`).
- The first request loads the model (SD cold load ~1–3 min), later ones use it.
- Switching model/type = unload of the previous one (single-resident).
- After `UNIFIED_IDLE_UNLOAD_SECONDS` (default 900 s) without requests the model
  is released; during an in-flight request it is **never** unloaded
  (in-flight guard — otherwise a cold load > limit would kill its own generation).
- Diagnostics: `/health` → `idle_seconds`, `idle_unload_seconds`, `in_flight`.

## FFmpeg tools (WebUI Tools tab + CLI)

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

## Switching the Windows service to unified

The `OdysseusDiffusion` service currently runs the old `pc/diffusion_server.py`. Switching
(changes the running service, run it deliberately):

```powershell
.\Switch-Service-ToUnified.ps1 -WhatIf   # shows what it would change
.\Switch-Service-ToUnified.ps1           # switches to unified_server
.\Switch-Service-ToUnified.ps1 -Revert   # reverts to the old diffusion_server
```

## Odysseus wiring

- Image base URL: `http://<host>:8101`, audio base URL: `http://<host>:8101`
  (paths `/v1/audio/...` identical to the former :4444).
- Contract: `pc/docs/UNIFIED-CONTRACT.md` (frozen). The old contract still applies:
  `docs/ODYSSEUS-CONTRACT.md`.

## Portable package (Win/Mac/Linux)

V1 = repo + `.venv` + `models/` + `Start-Unified.ps1` (Win) /
`uvicorn unified_server:app` (.venv python) everywhere. Mac: `--cpu` path
automatically (torch MPS/CPU, SD slow but functional); Linux: same venv
procedure as Win. A full installer (PyInstaller/Briefcase) only after V1
stabilizes — the venv is ~8 GB, models ~19 GB, none of it should be committed.

## Android

See `android/README.md` + `android/contract-compat.md` and `android/docs/ANDROID.md`.
`./gradlew :app:assembleDebug` only on a machine with the Android SDK.

The Android app is a **standalone offline server with its own native UI**
(tabs: Server / Generate / Transcribe / Upscale / Caption / Tools; models
manager; settings; console). There is **no WebUI in the APK**, no
Local/Auto/Remote switch and no remote host — every request is served locally.

Working on-device: `/health`, `/v1/models`, `/v1/audio/*` (whisper JNI),
`/v1/models/catalog|install|uninstall|cancel|add|delete`,
`/v1/images/generations` (ONNX LCM QDQ), `/v1/settings`.
Edits/inpaint/harmonize/upscale/caption/chat/ffmpeg return 503 until the
on-device engines land (see `docs/ONDEVICE-PLAN.md`).

### Test via adb

```powershell
powershell -ExecutionPolicy Bypass -File android/tools/monitor_android.ps1
```
Script: forward `8199→8101`, health, models, transcription smoke, resource
snapshot, logcat dump.
