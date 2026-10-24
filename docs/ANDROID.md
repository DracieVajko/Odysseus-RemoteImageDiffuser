# Android — the phone as a Prism server

The phone is a **standalone, offline AI server**. No WebUI, no PC/remote mode.
Every feature runs locally; models are downloaded once and then used offline.

## Features

| Feature | Status | Engine |
|---|---|---|
| Transcribe / Translate | works | whisper.cpp (JNI), 30 s auto-split chunks |
| Generate (txt2img) | works | ONNX Runtime, LCM-Dreamshaper QDQ UNet |
| Edits (img2img) | planned | ONNX Runtime (VAE encoder already downloaded) |
| Inpaint | planned | ONNX Runtime + latent mask blend |
| Harmonize | planned | CPU (LAB color match + feather) |
| Upscale | planned | Real-ESRGAN ONNX or CPU fallback |
| Caption / Chat | experimental | SmolVLM (llama.cpp runtime not bundled yet) |
| FFmpeg tools | planned | bundled ffmpeg binary |

See `docs/ONDEVICE-PLAN.md` for the implementation plan and order of work.

## UI

Native Android app with a tab bar: **Server / Generate / Transcribe / Upscale /
Caption / Tools**. Tabs can be tapped or swiped (slide + fade animation).

- **Server**: status, START/STOP, models manager, settings, console.
- **Generate**: prompt, size (default 512×512), steps (default 6), preview, Save image.
- **Transcribe**: file picker, audio-model spinner, language, transcript, Save transcript.
- **Upscale / Caption**: image picker, action, output.
- **Tools**: FFmpeg note (experimental).
- Console: scrollable, selectable, copy/clear.
- Models: collapsible sections by kind, install/uninstall, progress, **Stop N%**
  to cancel a download, **+ Add model by URL**.

## Model management

The catalog is in `ModelCatalog.kt` — **every URL is verified** (HTTP 200 +
Content-Length) and every model has an exact file list. Directory-based HF
repositories are downloaded file by file.

| Group | Examples | Engine |
|---|---|---|
| Transcription | whisper-large-v3-turbo q5_0 (547 MB) | whisper.cpp (local) |
| Image generation | LCM-Dreamshaper v7 INT8 ONNX → QDQ (1.6 GB) | onnxruntime |
| Upscale | Real-ESRGAN x4plus NCNN (45 MB zip) | vulkan-ncnn (experimental) |
| Vision-language | SmolVLM 500M / SmolVLM2 500M GGUF + mmproj | llama.cpp (experimental) |
| Custom | any direct URL (GGUF/ONNX/bin/safetensors) | depends on type |

Downloads are handled by `ModelDownloader` (own engine, not DownloadManager):

- resume via HTTP `Range` (`<file>.part`),
- staging `models/.staging/<id>/` → move after completion,
- `.READY` JSON manifest with sizes + SHA-256 (if present),
- progress via API and UI, cancel at any time,
- legacy `.READY` markers from the DownloadManager era are migrated automatically.

## API (OpenAI-compatible)

| Endpoint | Description |
|---|---|
| `GET /health` | status + installed models |
| `GET /v1/models` | list + `kinds` |
| `GET /v1/models/catalog` | catalog with status, `install_progress`, `engine`, `files` |
| `POST /v1/models/install` | `{"id": "..."}` → download in the background |
| `GET /v1/models/progress/{id}` | state/percent/bytes/file_index |
| `POST /v1/models/cancel` | cancels the download |
| `POST /v1/models/uninstall` | deletes weights + manifest |
| `POST /v1/models/add` | `{"name","kind","url"}` → custom model |
| `GET /v1/models/custom` | list of custom models |
| `POST /v1/models/delete` | deletes a custom model |
| `POST /v1/audio/transcriptions` | local whisper.cpp |
| `POST /v1/audio/translations` | local whisper.cpp |
| `POST /v1/images/generations` | local ONNX LCM |
| `POST /v1/images/edits|inpaint|harmonize|upscale|caption` | 503 until implemented |
| `POST /v1/chat/completions` | 503 until the VLM runtime lands |
| `GET /v1/ffmpeg/presets`, `POST /v1/ffmpeg/{op}` | 503 until ffmpeg is bundled |
| `GET /v1/settings` | runtime info |

The server binds `0.0.0.0:8101`, so any external AI client on the LAN can use it.

## Single-resident models

Only one heavy engine is loaded at a time:

- image generation calls `WhisperLib.release()` before loading the LCM engine,
- audio calls `LcmOnnxEngine.release()` before loading whisper,
- `LcmOnnxEngine.release()` runs after every generation,
- `ModelManager` enforces the switch (503 `loading model` during cold load).

## Hardware adaptation

- **Threads**: `Runtime.availableProcessors()` clamped to [1, 8], `largeHeap` enabled.
- **Image size**: default 512×512; `HardwareProfile` can reduce it on low-RAM devices.
- **Engines**: each model declares `engine`; unsupported engines return 503.

## Known blockers / experimental

- **VLM (SmolVLM)**: the llama.cpp/mtmd runtime is not bundled yet. The endpoint
  returns 503. Plan: ONNX SmolVLM or llama.cpp integration.
- **Upscale**: Real-ESRGAN NCNN engine not implemented; CPU fallback planned.
- **FFmpeg**: no ffmpeg binary on Android; bundled binary planned.
- **Edits / Inpaint / Harmonize**: planned (see `docs/ONDEVICE-PLAN.md`).

## Image generation note

The INT8 LCM model uses `ConvInteger(10)`, which ORT Android does not implement.
The UNet was converted to **QDQ** (`convert_lcm_qdq.py`); the phone runs the
converted inline model (824 MB). 512×512 / 6 steps ≈ 60–70 s; 256×256 / 4 steps
≈ 20–30 s.

## Licenses

- LCM-Dreamshaper: OpenRAIL.
- Real-ESRGAN BSD-3, Waifu2x MIT.
- ffmpeg-kit only Min/HTTPS (LGPL) — Full/Main = GPL.
- Models are downloaded lazily via ModelDownloader (SHA-256 + `.READY`), never in the APK.
