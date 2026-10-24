# Prism Checklist

Status legend: [x] done, [~] in progress, [ ] todo, [!] experimental/blocked.

## Architecture (locked 2026-10-22)

- [x] Android = native app only, **no WebUI**.
- [x] Android = **no PC/remote mode** — Local/Auto/Remote removed, remote host
      removed, `ProxyForward` deleted.
- [x] Every Android feature runs locally and offline.
- [x] PC = separate server + WebUI.
- [x] Both expose OpenAI-compatible endpoints on LAN.
- [x] Settings in the native app.
- [x] Models in the native app.
- [x] Unload model after each request.

## Android native app

- [x] Tabs: **Server / Generate / Transcribe / Upscale / Caption / Tools**.
- [x] Active tab highlighted (blue, white text); inactive gray.
- [x] Swipe left/right with slide+fade animation.
- [x] Server: status, models manager, settings, console (mode card removed).
- [x] Generate: prompt, size, steps, preview.
- [x] Transcribe: file picker, audio-model spinner, language, output.
- [x] Upscale: image picker, 2x button.
- [x] Caption: image picker, caption button.
- [x] Tools: ffmpeg note.
- [x] File pickers via `ActivityResultContracts.OpenDocument` (`*/*` for audio).
- [x] All UI text in English.
- [x] Keyboard does not auto-open.
- [x] Max audio size limit removed (chunking handles it).
- [x] Console log: scrollable (220 dp) + selectable + **copy** button.
- [x] **Save image** button (Pictures/Prism via MediaStore).
- [x] **Save transcript** button (Download/Prism via MediaStore).
- [x] Download progress button says **Stop N%** and cancels the download.
- [x] Single-resident: audio releases the image engine, image releases whisper.

## On-device features

- [x] Edits (img2img) — ONNX LCM VAE encoder, wired in `ApiServer`.
- [x] Inpaint — add/remove/color modes, ONNX LCM, wired in `ApiServer`.
- [x] Harmonize — CPU fallback in `ImageOps.kt`, wired in `ApiServer`.
- [ ] Upscale — NCNN Vulkan engine skeleton (`RealEsrganNcnnEngine.kt`), JNI placeholder; 503 until `librealesrgan_jni.so` built.
- [ ] Caption / Chat — llama.cpp (mtmd) engine skeleton (`SmolVlmEngine.kt`), JNI placeholder; 503 until `libsmolvlm_jni.so` built.
- [ ] FFmpeg — plan §6 (bundled binary).

## Android endpoints

- [x] GET /health, /v1/models, /v1/models/catalog, /v1/models/custom
- [x] POST /v1/models/install, /uninstall, /cancel, /add, /delete
- [x] GET /v1/models/progress/{id}
- [x] POST /v1/images/generations (local ONNX) — **200**
- [x] POST /v1/images/edits (img2img) — ONNX LCM, **200**
- [x] POST /v1/images/inpaint (add/remove/color) — ONNX LCM, **200**
- [x] POST /v1/images/harmonize — CPU, **200**
- [ ] POST /v1/images/upscale — NCNN skeleton wired, JNI placeholder, **503**
- [ ] POST /v1/images/caption — llama.cpp skeleton wired, JNI placeholder, **503**
- [ ] POST /v1/chat/completions — llama.cpp skeleton wired, JNI placeholder, **503**
- [x] POST /v1/audio/transcriptions, /translations — local whisper.cpp
- [ ] GET /v1/ffmpeg/presets / POST /v1/ffmpeg/{op} — 503 until ffmpeg lands
- [x] GET /v1/settings
- [x] Bind 0.0.0.0 (LAN reachable)
- [x] /webui removed

## PC (unified server)

- [x] FastAPI server with all endpoints.
- [x] Separate WebUI with all tabs.
- [x] Model catalog + install/uninstall + custom URL.
- [x] FFmpeg presets (5 video, 6 audio, 8 resize).
- [x] Bind 0.0.0.0.
- [x] Dropdown alignment + hideable missing-models message.
- [x] Console: scrollable (220 px), selectable, **copy** + **clear**.
- [x] **Save transcript** link (downloads `.txt`).
- [x] **Stop** button for model downloads (real cancel).
- [x] Image generation works without torch/diffusers via ONNX LCM fallback
      (`unified/lcm_onnx_numpy.py`).
- [x] Cancellation: URL downloads stop in the chunk loop, HF snapshots between
      files.

### Image generation (2026-10-23)

- [x] Positive prompt + **negative prompt** (default editable).
- [x] **Guidance** field (1.0 = LCM; >1 enables CFG with the negative prompt).
- [x] **Resolution dropdown** (256/512/768) + **custom WxH**.
- [x] **Quality dropdown** (low/medium/high) + explicit steps.
- [x] PC: ONNX LCM CFG (`unified/lcm_onnx_numpy.py`).
- [x] Android: ONNX LCM CFG (`LcmOnnxEngine.generate`).
- [x] Inpaint: image + mask + mode (add/remove/color) + strength.
- [x] Color mode: recolor the masked region (no AI, preserves luminance).
- [x] PC img2img (`/v1/images/edits`) uses the ONNX VAE encoder.
- [x] Android img2img/inpaint wired in `ApiServer` + `LcmOnnxEngine`.
- [x] Android harmonize (CPU) in `ImageOps.kt` + `ApiServer`.
- [ ] Android device verification (pending reconnect).

### FFmpeg presets (2026-10-23)

- [x] Video: 10 presets (H.264/H.265 HQ/fast, Discord <10 MB, GIF, WebM).
- [x] Audio: 8 presets (MP3 128/192/320, AAC 128/192, WAV 16k, Opus, FLAC).
- [x] Image: 5 presets (JPG 95/85, PNG, WebP, AVIF) + `/v1/ffmpeg/convert-image`.
- [x] WebUI Tools tab uses the presets dynamically.

### Test data (2026-10-23)

- [x] `testdata/images/test_512.png`, `mask_512.png`
- [x] `testdata/audio/jfk.wav`, `tone_3s.wav`
- [x] `testdata/video/test_5s.mp4`
- [x] `testdata/prompts.txt`

### PC test result (2026-10-24, CUDA)

`test_pc_features.py`: **31/31 passed** — health/models/catalog/settings,
txt2img (negative + guidance), custom size, img2img, inpaint add/remove/color,
convert-image/audio/video, extract-audio, trim, info, compress, transcription,
translation. GPU (RTX 3060) warm timings: 256/4 3–6 s, 384/6 ~9–11 s,
512/6 ~52 s.

`test_ffmpeg_presets.py`: **23/23 passed** — all 10 video, 8 audio and 5 image
presets.

### Android on-device test (2026-10-23)

Verified on Vivo V2415 (SD 8 Gen 2, 12 GB), final build:

| Test | Result |
|---|---|
| Install + launch + settings (Theme/Language/Acceleration) | OK |
| `GET /health`, `/v1/models`, `/v1/models/catalog`, `/v1/settings` | 200 |
| Generate 256×128, n=2 | 200, `size=256x256`, `requested_size=256x128`, 2 distinct images |
| Generate 384×384 (negative prompt, g=2) | 200, ~48 s |
| Edits (img2img, VAE encoder) | 200 |
| Inpaint add | 200, inside 72.8 / outside 0.0 (mask respected) |
| Inpaint remove | 200, engine=onnx |
| Inpaint color (non-AI) | 200, engine=onnx |
| Harmonize (mask) | 200, engine=cpu-harmonize |
| Transcribe | 200, whisper-large-v3-turbo |
| Caption / Upscale / FFmpeg | 503 (expected) |

PID stayed 2390 across generate, inpaint add/remove/color, harmonize,
transcription and another generation — no process restart after the mask
broadcast fix.

**Mask polarity (Android):** black 0.0, white 145.0, gray128 62.1,
alpha128 62.1 mean absolute diff — correct polarity and partial masks.

**State reporting (Android):** during a 512×512 generation `/health` reported
`busy=true`, `loading=true` during engine init, then `loading=false` until
completion; after it `busy=false, queue=0`.

**Load test (Android):** 20 mixed requests (generates + inpaint color +
harmonize) → **20/20 OK**, PID unchanged (12393), threads 55–56, RSS 1.9–3.9 GB,
ending 2.66 GB. No OOM, no restart.

**PC concurrency:** `test_model_manager_concurrency.py` PASS — switch releases
the old engine before the new state is visible, concurrent ensures do not
deadlock, unload callbacks match switches.

**PC regressions:** `quality=high` → 8 steps, `n=2` distinct images,
`size`/`requested_size` reported, inpaint mask polarity black 0.0 / white 86.4 /
gray 35.1.

### Audit fixes (2026-10-23)

- [x] Android inpaint composite uses `alpha * luminance` (mask respected).
- [x] PC `_denoise` accepts the sliced timestep schedule for img2img/inpaint.
- [x] Android non-square size: square canvas + `size`/`requested_size` in the
      response; `n` generates distinct images.
- [x] PC `ensure` switch calls `on_unload` after the lock.
- [x] Android `ModelManager` no longer calls `System.gc()` under the lock;
      comments translated.
- [x] Removed leftover `android/app/src/main/assets/webui.html`.
- [x] PC `quality=high` → 8 steps; `n` uses distinct seeds.
- [x] WebUI inline colors use CSS variables (light theme complete).
- [x] Remaining Slovak comments translated (ApiServer, AudioDecoder, WhisperLib,
      ModelManager, ffmpeg_tools, registry, vlm, ytdlp, unified_server).
- [x] Repo cleanup: 109 scratch/test files removed (128 → 29 untracked).

### Harmonize (2026-10-23)

- [x] PC `/v1/images/harmonize` now uses CPU color matching + seam feather
      (`engine=cpu-harmonize`). Without a mask it applies a mild auto-level.
- [x] Android harmonize (`ImageOps.kt`, `engine=cpu-harmonize`).

### Acceleration (2026-10-23)

- [x] PC: ONNX providers auto-select CUDA/DirectML when available, CPU fallback.
      CPU sessions use `cpu_count-1` threads.
- [x] Android: acceleration setting **Auto / CPU / GPU (NNAPI)**:
      - Auto → XNNPACK first, NNAPI fallback;
      - GPU → NNAPI first, XNNPACK fallback;
      - CPU → no GPU providers.
- [x] Changing acceleration releases the engine; the next request reloads it.

### Themes and language (2026-10-23)

- [x] PC WebUI: dark/light toggle in the header, persisted in localStorage,
      CSS variables for all main colors.
- [x] Android: Theme setting **System / Light / Dark**
      (`AppCompatDelegate.setDefaultNightMode`), color resources in
      `values/colors.xml` + `values-night/colors.xml`.
- [x] Android: Language setting **English / Slovak / Czech / German**
      (`AppCompatDelegate.setApplicationLocales`), base strings English in
      `values/strings.xml`, translations in `values-sk`, `values-cs`, `values-de`.
- [x] Default language is English; default theme is System (follows the phone).

Note: MP3 192/320 produce the same size for a 16 kHz mono source because
libmp3lame caps the bitrate at that sample rate; use 44.1/48 kHz for higher
bitrates.

## Test suite

`python -m pytest tests -q` — 44 passed (2026-10-23).

`test_all_endpoints.py` (repo root) — 25 checks against the Android server.
Last result: **25/25 passed** (before the PC-mode removal).

| Area | Result |
|---|---|
| health, models, catalog, custom, settings, progress | 200 |
| transcribe / translate | 200 |
| generate 256×256 (~29 s) / 512×512 (~70 s) | 200, valid PNG |
| edits/inpaint/harmonize/upscale/caption/chat/ffmpeg | 502 before removal; now 503 |
| /webui | 404 |
| validation | 400 |

### Image quality (2026-10-23)

- Verified the Kotlin pipeline against a standalone numpy/onnxruntime reference
  (`lcm_python_ref.py`). Stats match closely (mean/std/unique/sharpness), so the
  implementation is correct.
- 4-step LCM at 256×256 is inherently soft/blurry; the model was trained at
  512×512. Default in the UI is now **512×512 / 6 steps**.
- SD 1.5 ONNX removed from the Android catalog (the LCM engine cannot run it;
  4.1 GB download with no engine).
- VLM badge changed from "PC only (Remote)" to "llama.cpp (experimental)".

### UI fixes (2026-10-23)

- Console: scrollable (220 dp), selectable, **copy** button.
- Save image (Pictures/Prism) and Save transcript (Download/Prism).
- Download button shows **Stop N%** and cancels.
- File picker: OpenDocument with initial URI `Recordings` (audio) / `Pictures`
  (images) and `*/*` for audio, so `.wav` files are visible.

## Docs

- [x] docs/ANDROID.md
- [x] docs/HANDOFF.md
- [x] docs/CHECKLIST.md (this file)
- [x] docs/CHECKPOINTS.md
- [x] docs/ONDEVICE-PLAN.md
- [x] docs/ARCHITECTURE.md (end-to-end PC + Android)
- [x] AGENTS.md

## English only (2026-10-23)

- [x] All Kotlin comments, docstrings, UI strings, log messages and model notes
      translated to English (0 Slovak diacritics in `android/**/*.kt`).
- [x] All Python comments/docstrings/messages translated to English
      (`unified_server.py`, `download_models.py`, `unified/*.py`,
      `diffusion_server.py`, `unified_shim.py`).
- [x] WebUI (`webui_unified.html`) is English-only.
- [x] Android layout XML and themes comments are English.
- [x] `RemoteProxy.kt` deleted (remote mode removed).
- [x] Docs translated to English (ANDROID, HANDOFF, UNIFIED-CONTRACT, UNIFIED-OPS).
- [x] `android/README.md` and `android/contract-compat.md` translated to English.
- [x] Final scan: **0 Slovak diacritics** in Kotlin, Android XML, Python,
      WebUI, docs, AGENTS.md.

## Cleanup

- [x] Remove scratch fix_*.py / check_*.py files from repo root (109 removed on
      2026-10-23; `test_lcm_tmp.py` and `test_android_continue.py` removed later).
- [x] Test scripts write sample outputs to a temp directory, not the repo root.
- [x] `ffmpeg2pass-0.log` no longer written to the server working directory.

## State reporting (2026-10-23)

| Check | Result |
|---|---|
| PC `/health` idle | `busy=false, queue=0, in_flight=0` |
| PC one request | `busy=true, in_flight=1, queue=0` |
| PC two concurrent | `busy=true, queue=1, in_flight=2`, both 200 |
| PC after completion | `busy=false, queue=0, in_flight=0` |
| PC after 400 | state back to idle |
| PC unit test | `test_begin_end_updates_busy_and_queue` PASS |
| Android one request | `busy=true, loading=true` during init, then `loading=false`, after `busy=false, queue=0` |
| Android two concurrent | `busy=true, queue=1` (verified) |
| Android completion under concurrency | **not verified** — device detached mid-test |

Test files: `test_pc_states.py` (PASS) and `test_android_states.py`
(queue verified; completion re-run pending when the phone is attached).

`test_android_features.py` now also asserts the Android regressions
(`n=2` → 2 distinct images, `size=256x256` + `requested_size=256x128` for a
`256x128` request); the run is pending until the device is reattached.

## Size reporting (2026-10-23)

- Both platforms use a square canvas sized by the **larger** requested side,
  clamped to 256–512.
- Responses report `size` (actual) and `requested_size` (as asked).
- Verified: `320x256` → `size=320x320`, `requested_size=320x256` (PC);
  `256x128` → `size=256x256`, `requested_size=256x128` (Android).

## GPU (PC, 2026-10-24)

- [x] `onnxruntime-gpu 1.31.0` + `nvidia-cudnn-cu13` installed.
- [x] `unified/onnx_env.py`: DLL registration + provider selection
      (`auto` → CUDA → CPU; DML skipped for QDQ).
- [x] `/health` reports `device` and `onnx_provider`.
- [x] `release_pipeline()` frees ONNX sessions on idle/model switch.
- [x] Measured (warm): 256/4 6.3 s, 384/6 10.7 s, 512/6 51.8 s (CPU: 23/45/65 s).
- [x] First run of a new shape pays cuDNN planning (43/65/184 s, one-time).
- [x] `PRISM_ONNX_PROVIDER=cpu` documented as the fallback for the 6 GB card.
- [ ] AMD iGPU: DirectML cannot run the INT8 QDQ graph → CPU only; needs a
      non-QDQ FP16/FP32 model in a separate `onnxruntime-directml` env.
