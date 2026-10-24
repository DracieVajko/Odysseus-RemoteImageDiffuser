# Prism — Architecture and Operation Guide

This document explains how Prism works end to end on the **PC server** and on the
**Android app**: processes, components, request lifecycles, model handling,
acceleration, UI, and the differences between the two.

---

## 1. Overview

Prism is **two independent products** that share one API contract:

| Product | Runtime | UI | Dependencies |
|---|---|---|---|
| **PC server** | Python 3 + FastAPI + Uvicorn | WebUI (`webui_unified.html`) | optional torch/diffusers, onnxruntime, faster-whisper, ffmpeg |
| **Android app** | Kotlin + Ktor + ONNX Runtime + whisper.cpp | native app (no WebUI) | none at runtime; models downloaded lazily |

Both expose an **OpenAI-compatible HTTP API** on port `8101`, so external clients
(OpenAI SDK, Odysseus, scripts) can attach over LAN.

Core principles:
- **Local and offline**: after model download, no network is needed.
- **Single resident model**: at most one heavy engine loaded at a time.
- **Honest errors**: unavailable features return `503 will-not-load` with a reason.
- **Lazy models**: weights are never bundled; they are downloaded on demand.

---

## 2. Repository map

```
unified_server.py              PC FastAPI app (routes, lifespan, middleware)
webui_unified.html             PC WebUI (tabs, console, model manager, themes)
unified/
  engine.py                    diffusion_server init + inpaint bridge
  lcm_onnx_numpy.py            ONNX LCM pipeline (txt2img/img2img/inpaint/harmonize)
  model_manager.py             single-resident gate + idle unload
  models_manager.py            PC install/uninstall/state (download_models.py)
  model_finder.py              HF search + custom URL add/delete
  upscale.py                   Real-ESRGAN (torch) or Pillow fallback
  vlm.py                       SmolVLM caption/chat (transformers)
  audio_proxy.py               forward to WhisperApi :4444
  audio_splitter.py            30 s chunking for long audio
  ffmpeg_tools.py              presets + operations
  ffmpeg_pre.py                normalize audio/video for whisper
  ytdlp.py                     download audio from URL
download_models.py             PC model download tasks (snapshot/file/url)
models/                        PC weights (not committed)
testdata/                      sample images/audio/video/prompts
android/
  app/src/main/java/eu/odysseus/diffusion/
    MainActivity.kt            UI (tabs, actions) + DiffusionService
    ApiServer.kt               Ktor routes + request/response contract
    LcmOnnxEngine.kt           ONNX LCM pipeline + inpaint/harmonize
    ImageOps.kt                CPU harmonize
    ModelCatalog.kt            verified model catalog
    ModelDownloader.kt         download engine (resume, manifest, cancel)
    ModelManager.kt            single-resident gate
    WhisperLib.kt              whisper.cpp JNI wrapper
    AudioDecoder.kt            MediaCodec/WAV -> 16 kHz PCM
    AppConfig.kt               acceleration setting
```

---

## 3. API contract (both platforms)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, loaded model, available models |
| GET | `/v1/models` | model list + kinds |
| GET | `/v1/models/catalog` | catalog + install state + engine |
| POST | `/v1/models/install` | start download `{"id": ...}` |
| GET | `/v1/models/progress/{id}` | download progress |
| POST | `/v1/models/cancel` | stop download |
| POST | `/v1/models/uninstall` | delete weights |
| POST | `/v1/models/add` | custom model (PC: `repo_id`; Android: `url`) |
| GET | `/v1/models/custom` | custom models |
| POST | `/v1/models/delete` | delete custom model |
| POST | `/v1/images/generations` | txt2img |
| POST | `/v1/images/edits` | img2img (multipart) |
| POST | `/v1/images/inpaint` | inpaint with mask |
| POST | `/v1/images/harmonize` | color match + seam fix |
| POST | `/v1/images/upscale` | upscale |
| POST | `/v1/images/caption` | VLM caption |
| POST | `/v1/chat/completions` | VLM chat |
| POST | `/v1/audio/transcriptions` | speech to text |
| POST | `/v1/audio/translations` | speech to English |
| GET | `/v1/ffmpeg/presets` | preset catalog |
| POST | `/v1/ffmpeg/{op}` | convert/trim/compress/extract/info |
| GET | `/v1/settings` | runtime info |

Common JSON shapes:
- `generations`: `{prompt, negative_prompt, guidance, size, quality, n, steps, request_id}`
- `inpaint`: `{image, mask, prompt, negative_prompt, guidance, mode, color, strength, steps, width, height}`
- error: `{"error": "...", "reason": "...", "needed_mb": -1, "free_mb": -1}`

---

## 4. PC server

### 4.1 Process and lifespan

`unified_server.py` runs under Uvicorn. At startup:
1. `init_diffusion_engine()` initializes the optional torch/diffusers engine
   lazily (registry only, no weights).
2. A background task `_idle_unloader()` runs every 30 s and unloads the resident
   model after `UNIFIED_IDLE_UNLOAD_SECONDS` (default **900 s**) without requests.
3. Middleware `_track_inflight` increments/decrements an in-flight counter for
   `/v1/images/*`, `/v1/audio/*`, `/v1/chat/*`, so the idle unloader cannot evict
   a model during a long generation.

### 4.2 Single-resident ModelManager

`unified/model_manager.py`:
- `ensure(kind, id)` — if a different model is resident, unload it first, then
  report `loading` (503) while the caller loads weights.
- `begin()/end()` — in-flight counter.
- `idle_seconds()`, `status()` — used by `/health` and the idle unloader.

Rules:
- Image models (SD/SDXL) and audio (faster-whisper) share VRAM → never both.
- Calls to `_release_engines()` unload torch pipelines and free VRAM.

### 4.3 Image generation

Two backends, tried in order:

**A. torch/diffusers (if installed)**
`_try_real_generation()` imports `diffusion_server`, resolves the model, calls
`ensure_model_loaded()` + `_generate_locked()`. Uses CUDA when available,
otherwise CPU. Supports SD 1.5 / SDXL / SD-Turbo and negative prompts natively.

**B. ONNX LCM fallback (no torch)**
`unified/lcm_onnx_numpy.py` runs the same model as Android:
1. **Tokenizer** — CLIP BPE from `tokenizer/vocab.json` + `merges.txt`.
2. **Text encoder** `text_encoder/model.onnx` → `(1,77,768)`.
3. **Scheduler** — LCM scaled-linear betas; timesteps for N steps
   (`[999, 759, 499, 259]` for 4 steps).
4. **UNet** `unet/model_qdq_inline.onnx` (QDQ) with `sample`, `timestep`,
   `encoder_hidden_states`, `timestep_cond`.
   - **CFG**: if `guidance > 1` and a negative prompt is set, the UNet runs twice
     and `noise = neg + g * (pos - neg)`; `timestep_cond` encodes `g - 1`.
5. **VAE decoder** `vae_decoder/model.onnx` → RGB.
6. Size is capped to **256–512 px** (the model is trained at 512); steps: 6 for
   medium, 4 for low.

**Providers**: CUDA → DirectML → CPU auto-selection. With a GPU provider, ORT
places supported nodes on the GPU and the rest on the CPU. CPU sessions use
`cpu_count - 1` threads. (The installed `onnxruntime` wheel is CPU-only; install
`onnxruntime-gpu` for CUDA.)

### 4.4 img2img, inpaint, harmonize

- **img2img** (`/v1/images/edits`): VAE encoder `vae_encoder/model.onnx` →
  latents; start at `round((1-strength)*steps)`; add noise at that timestep;
  denoise; VAE decode. Returns `data[0].b64_json`.
- **inpaint** (`/v1/images/inpaint`):
  - `mode=add|remove`: VAE encode → noise → denoise with the prompt; at each step
    `latents = mask*latents + (1-mask)*original`; decode; composite with a
    feathered mask.
  - `mode=color`: **non-AI** recolor of the masked region (HSV hue/saturation
    replaced, original luminance kept) — e.g. black → blue.
- **harmonize**: CPU only. With a mask, computes inside/outside mean colors and
  shifts the masked region toward the background, then feathers the seam.
  Without a mask it applies a mild auto-level. `engine=cpu-harmonize`.

### 4.5 Upscale, caption, chat

- **upscale** (`unified/upscale.py`): Real-ESRGAN x4 (RRDBNet) when weights +
  torch exist, tiled (256 px tiles, 16 px overlap); otherwise Pillow LANCZOS.
  scale 2/3 downsample the x4 result.
- **caption/chat** (`unified/vlm.py`): SmolVLM-500M via transformers; without
  weights/torch returns `503 will-not-load`.

### 4.6 Audio

- `/v1/audio/transcriptions`:
  - if `faster-whisper` is installed → local transcription (30 s chunks via
    `audio_splitter.py` when enabled);
  - otherwise → proxy to the legacy WhisperApi at `UNIFIED_WHISPER_URL`
    (default `http://127.0.0.1:4444`);
  - otherwise → `503` with an explicit stub message.
- `yt-dlp` URLs are downloaded first when a `url` form field is sent.

### 4.7 FFmpeg

`unified/ffmpeg_tools.py` provides:
- **10 video presets**: H.264/H.265 (default/HQ/fast), mp4-small,
  mp4-discord (<10 MB), webm, gif.
- **8 audio presets**: mp3 128/192/320, aac 128/192, wav-16k, opus, flac.
- **5 image presets**: jpg-95, jpg-85, png, webp, avif.
- Operations: convert-video/audio/image, extract-audio, trim, compress, info.

The WebUI Tools tab loads `/v1/ffmpeg/presets` dynamically and sends the chosen
preset.

### 4.8 Model downloads (PC)

`download_models.py` tasks are `snapshot` (HF repo), `file` (single HF file) or
`url` (direct). `models_manager` runs installs in a thread and exposes state.
Cancellation uses `_CANCEL` + `CancelledError`; URL downloads stop in the chunk
loop, HF snapshots stop between files (per-file `hf_hub_download`).

### 4.9 WebUI

`webui_unified.html`:
- Tabs: Text-to-image, Image-to-image, Inpaint (canvas mask), Upscale,
  Transcribe, Caption, Tools, Models, Settings.
- Positive + negative prompt, guidance, resolution (256/512/768/custom), quality.
- Console: scrollable, selectable, copy/clear.
- Models: catalog grouped by kind, install/uninstall, **Stop** for downloads.
- Dark/light theme toggle (CSS variables, localStorage).

---

## 5. Android app

### 5.1 Process model

- `MainActivity` hosts a native UI with a horizontal tab bar:
  **Server / Generate / Transcribe / Upscale / Caption / Inpaint / Tools**.
- Tabs can be tapped or swiped (slide + fade).
- `DiffusionService` is a **foreground service** that owns the Ktor server, so it
  keeps running when the app is backgrounded.
- The server binds `0.0.0.0:8101`; `adb forward tcp:8199 tcp:8101` maps it to the
  PC for testing.

### 5.2 Server (Ktor)

`ApiServer.kt` installs:
- `ContentNegotiation` with **Gson** (responses use Gson tree types; requests use
  kotlinx `wireJson` for defaults + unknown keys).
- `StatusPages` mapping `ModelLoadingError` → 503 loading and
  `ModelWillNotLoadError` → 503 will-not-load.
- Routes: health/models/catalog/install/cancel/uninstall/add/delete,
  images generations/edits/inpaint/harmonize/upscale/caption, chat, audio
  transcriptions/translations, ffmpeg (503), settings.

No WebUI, no remote/PC mode: every request is served on the phone.

### 5.3 Model management

`ModelCatalog.kt` — verified URLs and exact file lists; `ModelDownloader.kt`:
- Resume via HTTP `Range` (`<file>.part`).
- Staging `models/.staging/<id>/` → atomic move on completion.
- `.READY` JSON manifest with sizes + optional SHA-256.
- Progress via API/UI; **Stop N%** cancels.
- Multi-file models (LCM ONNX has 7 files; SD 1.5 was removed because the LCM
  engine cannot run it).
- Custom models by URL are stored in `models/<name>/`.

`ModelManager.kt` enforces single-resident switching; the engines themselves
release memory:
- image generation calls `WhisperLib.release()` before loading the LCM engine;
- audio calls `LcmOnnxEngine.release()` before loading whisper;
- `LcmOnnxEngine.release()` runs after every image request.

### 5.4 Audio pipeline

1. `AudioDecoder.kt` decodes WAV (fast path) or any MediaCodec-supported format
   to **16 kHz mono FloatArray**.
2. `handleAudio` splits the PCM into **30 s chunks** when needed.
3. `WhisperLib` (JNI `libwhisper_jni.so`) runs whisper.cpp with the GGML model.
4. Text chunks are concatenated; the response includes `model`, `chunks`,
   `duration`, `auto_split`.

Default model: `whisper-large-v3-turbo-q5_0` (547 MB). If the requested model is
not installed, the smallest installed audio model is used.

### 5.5 Image pipeline

`LcmOnnxEngine.kt`:
- Loads `unet/model.onnx` (QDQ inline), `text_encoder`, `vae_decoder`,
  `vae_encoder` (for img2img/inpaint).
- CLIP BPE tokenizer, LCM scheduler, CFG with `timestep_cond`.
- `generate(prompt, negativePrompt, guidance, steps, size)`.
- `img2img(source, prompt, …, strength)`.
- `inpaint(source, mask, prompt, …, mode, color)`:
  - `add`/`remove`: VAE encode + mask blend per step + feathered composite.
  - `color`: HSV recolor of the masked region (no AI).
- `release()` closes all sessions and the environment.

Generation time on a Snapdragon 8 Gen 2: ~20–30 s for 256×256/4 steps,
~60–70 s for 512×512/6 steps (INT8 QDQ, CPU/XNNPACK).

### 5.6 Harmonize

`ImageOps.kt` (CPU): masked color match + seam feather; no-mask auto-level.
Returns `engine=cpu-harmonize`.

### 5.7 Settings

- **Theme**: System / Light / Dark (`AppCompatDelegate.setDefaultNightMode`),
  colors in `values/colors.xml` + `values-night/colors.xml`.
- **Language**: English (default) / Slovak / Czech / German
  (`AppCompatDelegate.setApplicationLocales`); base strings English, translations
  in `values-sk`, `values-cs`, `values-de`.
- **Acceleration**: Auto / CPU / GPU (NNAPI) — selects ONNX providers; changing
  it releases the engine so the next request reloads.
- Models, server status, console (scrollable/selectable/copy), Save image,
  Save transcript.

### 5.8 UI actions

- **Generate**: prompt, negative prompt, resolution (256/512/768/custom),
  quality, steps, guidance → `POST /v1/images/generations` → preview + Save.
- **Transcribe**: file picker (`Recordings`), audio model spinner, language,
  → transcript + Save.
- **Upscale**: image picker + 2× (engine is experimental; endpoint returns 503).
- **Caption**: image picker → caption (503 until the VLM runtime lands).
- **Inpaint**: image + mask + mode + color + prompt/negative/guidance/strength →
  preview + Save.
- **Tools**: FFmpeg note (no binary yet).

---

## 6. End-to-end flows

### 6.1 Text-to-image (PC)

```
WebUI → POST /v1/images/generations
      → ModelManager.ensure("image", id)
      → try torch/diffusers
      → fallback: LcmOnnxNumpy.generate(prompt, negative, guidance, size, steps)
         → text encoder → LCM loop (UNet ± CFG) → VAE decode
      → {"data":[{"b64_json":...}], "request_id":...}
```

### 6.2 Text-to-image (Android)

```
UI → POST http://127.0.0.1:8101/v1/images/generations
   → requireModelReady("lcm-dreamshaper-int8-onnx")
   → LcmOnnxEngine.init(appContext, modelDir)
   → generate(prompt, negative, guidance, steps, size)
   → Bitmap → base64 PNG → response
   → LcmOnnxEngine.release()
```

### 6.3 Inpaint (both)

```
image + mask + mode + prompt
→ VAE encode the image
→ mask to latent resolution (feathered)
→ add noise at strength timestep
→ denoise N steps, blend mask*latent + (1-mask)*original each step
→ VAE decode
→ composite over the original with the feathered mask
```

`mode=color` skips diffusion and recolors the masked region.

### 6.4 Transcription (Android)

```
file picker → bytes → AudioDecoder → 16 kHz FloatArray
→ 30 s chunks → WhisperLib.transcribe (JNI)
→ join text → response
```

### 6.5 Transcription (PC)

```
upload → faster-whisper (local, chunked) or WhisperApi :4444
→ text → response
```

---

## 7. Model lifecycle

1. **Catalog** — model id, kind, engine, files, sizes, verified URL.
2. **Install** — downloader writes to staging with resume, then to
   `models/<id>/`, then writes `.READY` (sizes + SHA-256).
3. **Load** — the engine loads only the requested model.
4. **Unload** — after the request (Android) or after the idle timeout / switch
   (PC).
5. **Uninstall** — deletes the model directory + manifest (path-guarded).

Android storage: `/storage/emulated/0/Android/data/eu.odysseus.diffusion/files/models/`.

---

## 8. Hardware adaptation

| Aspect | PC | Android |
|---|---|---|
| GPU | CUDA / DirectML via ONNX Runtime; torch CUDA if installed | NNAPI (GPU/NPU) via ORT; XNNPACK CPU |
| CPU | `cpu_count - 1` threads | `cores - 1`, max 8 |
| Memory | torch/ONNX sessions unloaded on idle/switch | `largeHeap`, engine released after each request |
| Image size | 256–512 ONNX fallback (torch can do more) | 256–512 (model limit) |
| Storage | `models/` | app external files `models/` |

---

## 9. PC vs Android differences

| Topic | PC | Android |
|---|---|---|
| UI | WebUI (browser) | native tabs |
| Server | FastAPI/Uvicorn | Ktor/Netty in a foreground service |
| Image backend | torch/diffusers, ONNX fallback | ONNX only (QDQ) |
| Inpaint | ONNX or diffusers | ONNX |
| Harmonize | CPU | CPU |
| Upscale | Real-ESRGAN (torch) / Pillow | stub (1×1 PNG; 503 if the model is not installed) |
| Caption/chat | SmolVLM (transformers) | 503 (VLM runtime not bundled) |
| Audio | faster-whisper / WhisperApi | whisper.cpp JNI |
| FFmpeg | full presets | 503 (no binary) |
| Models | HF snapshots + files + URLs | verified file lists + URLs |
| Themes | dark/light toggle | System/Light/Dark |
| Language | English | English/SK/CZ/DE |

---

## 10. Attaching external clients

Any OpenAI-compatible client can use either server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host>:8101/v1", api_key="not-needed")
client.images.generate(model="lcm-dreamshaper-int8-onnx", prompt="a red cube", size="512x512")
```

For audio:
```bash
curl -F "file=@audio.wav" -F "language=en" http://<host>:8101/v1/audio/transcriptions
```

---

## 11. Known limitations

- **Android VLM** (caption/chat): runtime not bundled → 503.
- **Android upscale**: Real-ESRGAN engine not implemented; the endpoint returns a
  1×1 stub PNG when the model is installed, 503 otherwise.
- **Android FFmpeg**: no binary → 503.
- **PC with CPU-only onnxruntime**: image generation is CPU-bound; install
  `onnxruntime-gpu` (CUDA) or `onnxruntime-directml` for GPU.
- **PC without torch**: SDXL/absolute reality/animagine/juggernaut cannot run;
  only the LCM ONNX model is available.
- **MP3 192/320** on a 16 kHz mono source are capped by libmp3lame (~160 kbps).

---

## 12. Run and build

**PC**
```bash
python -m uvicorn unified_server:app --host 0.0.0.0 --port 8101
# WebUI: http://<pc>:8101/webui
```

**Android**
```bash
cd android
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
# Start the server in the app (Server tab → START)
# API: http://<phone>:8101
```
