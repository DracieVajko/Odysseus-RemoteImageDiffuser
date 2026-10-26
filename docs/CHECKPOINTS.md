# Prism Checkpoints

Chronological record of decisions and verified states. Newest last.

## 2026-10-22 — Architecture pivot

**Decision:** Android = native app only (no WebUI). PC = separate WebUI with all
features. Both expose OpenAI-compatible endpoints on LAN. Android runs everything
locally; blocked items marked experimental with documented reasons.

**Why:** user wants two independent servers (phone and PC), each usable from any
AI client (OpenAI-compatible), without the phone depending on the PC.

## 2026-10-22 — Model download engine

- Replaced DownloadManager with a custom engine (HTTP Range resume, staging,
  `.READY` JSON manifest, multi-file, auto-unzip, cancel, progress).
- Verified: whisper-turbo 547 MB, LCM 1.6 GB, custom URL.
- Fixed legacy `.READY` migration.

## 2026-10-22 — Audio

- Kept only `whisper-large-v3-turbo-q5_0` (547 MB).
- Threads 4→8, `no_context`, no progress prints.
- Auto-split 30s chunks; verified 2-min (5 chunks) and 45-min (91 chunks).
- `largeHeap=true` fixes OOM on 45-min files.

## 2026-10-22 — LCM ONNX engine (blocked)

- Ported full pipeline to `LcmOnnxEngine.kt` (CLIP BPE tokenizer, text encoder,
  UNet, LCM scheduler, VAE decode, Bitmap).
- ORT Android 1.19.2/1.20.0 rejects `ConvInteger(10)` (CPU, NNAPI, XNNPACK).
- Result: `/v1/images/generations` returns 503 `LCM engine init failed`.
- Documented in android/docs/ANDROID.md; fallback to PC in Auto/Remote.

## 2026-10-22 — Rename + English

- "Odysseus Diffusion" → "Prism" (label, notification, header, docs).
- `AGENTS.md`: English only; architecture rules.

## 2026-10-22 — Remote proxy

- Per-endpoint forward before body parsing (JSON + multipart).
- Verified GET `/v1/ffmpeg/presets`, POST JSON and multipart edits via `adb reverse`.

## 2026-10-22 — Hardware adaptation

- `HardwareProfile`: threads 1-8, image size 512/384/256 by RAM.
- `largeHeap=true`.

## 2026-10-22 — UI

- Collapsible model sections (default closed, animated), ripple, count chip.
- "+ Add model by URL" dialog.
- Settings card in native app.
- WebChromeClient file chooser in WebView (will be removed with WebUI).

## 2026-10-22 — Native screens (no WebUI)

- Android: removed WebUIActivity, `/webui`, syncWebUi task, Open Web UI button.
- Added tab bar (HorizontalScrollView + MaterialButtons, no 5-item limit):
  **Server / Generate / Transcribe / Upscale / Caption / Tools**.
- Server: status, mode, models, settings, console.
- Generate: prompt/size/steps/preview.
- Transcribe: file picker + audio-model spinner + language + output.
- Upscale: image picker + 2x.
- Caption: image picker + caption.
- Tools: FFmpeg note (PC-only, experimental).
- Verified on device: app launches (69 nodes), tab switching works
  (Generate screen shows Prompt/256x256/4/GENERATE).
- Added active-tab highlight (blue #1F6FEB, white text) and swipe navigation
  (GestureDetector onFling, threshold 120 px, horizontal > vertical).
- Verified swipe left → Generate, swipe right → Server; health=200,
  transcription OK, Android `/webui` → 404 (removed).

## 2026-10-22 — Image generation works on-device

- Root cause: ORT Android does not implement `ConvInteger(10)` (QOperator INT8).
- Fix: converted the LCM UNet to QDQ format (`pc/tools/convert_lcm_qdq.py`):
  `DynamicQuantizeLinear + ConvInteger + Cast + Mul + Add` → `DequantizeLinear +
  Conv` (98/98 nodes converted, 0 skipped).
- Also fixed text encoder input: `input_ids` must be **int32** (was int64).
- Deployed inline QDQ UNet (824 MB) to the phone.
- Verified: `POST /v1/images/generations` → 200, valid 256×256 PNG,
  4 steps ≈ 19–22 s. Two consecutive generations OK.
- Model is released after generation (`LcmOnnxEngine.release()`).
- Proxy fix: if the PC is unreachable, image/upscale/caption now return
  **502** with a clear reason instead of 500 ("body already consumed").
- Upscale/caption remain stubs (502 when Auto/Remote, 503 in Local).

## 2026-10-22 — Tab highlight fix

- Bug: on first launch all tabs appeared blue.
- Cause: `selectTab(0)` returned early because `currentTab == 0` and
  `screenServer` was already visible, so the button tint was never applied.
- Fix: removed the early-return guard; use `backgroundTintList` (not
  `setBackgroundColor`) and re-apply after layout via `tabBar.post`.
- Verified by pixel sampling:
  first launch → Server `#1F6FEB`, others `#30363D`;
  after swipe → Generate `#1F6FEB`, others `#30363D`.

## 2026-10-22 — Full endpoint test

- `pc/tools/test_all_endpoints.py`: 25/25 checks passed on Android.
- Image gen: 256×256 ~29 s, 512×512 ~70 s, both valid PNGs.
- Audio: transcribe 200 (27 s), translate 200 (33 s).
- Source-dependent endpoints (edits/inpaint/harmonize/upscale/caption/chat/
  ffmpeg) → 502 with a clear message when the PC is unreachable.
- `/webui` → 404 (Android has no WebUI).
- Model management + validation (400) OK.
- PC server on :8200: all core endpoints 200; real image/audio backends
  unavailable in this Python env (no torch/faster-whisper), fallbacks work.

## 2026-10-22 — Android-only (PC option removed)

- Removed Local/Auto/Remote + remote host from the UI and code.
- Deleted `ProxyForward.kt`; no proxy logic in `ApiServer`.
- Everything is served on-device; unimplemented features return 503.
- Added: scrollable/selectable/copyable console, Save image, Save transcript,
  Stop download button, `OpenDocument` picker (`*/*` for audio).
- Single-resident: audio releases the image engine, image releases whisper.
- Plan for the remaining on-device features: `docs/ONDEVICE-PLAN.md`.

## 2026-10-23 — PC parity with Android

- WebUI console: scrollable (220 px), selectable, copy + clear.
- Transcript: Save transcript link (`.txt` download).
- Model downloads: real **Stop** button.
  - `pc/download_models.py` has `_CANCEL` + `CancelledError`.
  - URL downloads stop in the chunk loop; HF snapshots stop between files
    (per-file `hf_hub_download` instead of `snapshot_download`).
  - `models_manager.cancel` returns `cancelled: true`.
- Image generation without torch/diffusers: `pc/unified/lcm_onnx_numpy.py`
  (numpy + onnxruntime, same QDQ UNet). Wired as a fallback in
  `_try_real_generation`.
  - Verified: `POST /v1/images/generations` → 200, valid PNG
    (256×256 and 512×512).
- English text in ffmpeg presets and messages.

## 2026-10-23 — Prompts, guidance, inpaint, FFmpeg presets

- Image generation now accepts **negative prompt** + **guidance** (CFG when
  guidance > 1) on PC (`pc/unified/lcm_onnx_numpy.py`) and Android
  (`LcmOnnxEngine.generate`).
- Resolution: 256/512/768 + custom WxH; quality low/medium/high + steps.
- Inpaint:
  - PC: ONNX VAE encoder + latent mask blend + feathered composite.
  - Modes: `add`, `remove`, `color` (non-AI recolor, preserves luminance).
  - Android: `LcmOnnxEngine.inpaint` + new Inpaint tab (image/mask pickers,
    mode, color, prompt, negative, guidance, strength).
- img2img: PC `/v1/images/edits` uses the ONNX VAE encoder; Android
  `/v1/images/edits` calls `LcmOnnxEngine.img2img`.
- FFmpeg presets expanded: 10 video, 8 audio, 5 image; new
  `POST /v1/ffmpeg/convert-image`; WebUI Tools tab loads presets dynamically.
- Test data added under `testdata/` (images, mask, audio, video, prompts).
- PC test suite `pc/tools/test_pc_features.py`: **20/20 passed** (txt2img with
  negative/guidance, custom size, img2img, inpaint add/remove/color, all FFmpeg
  ops, transcription).
- Android build: `BUILD SUCCESSFUL`; device verification pending reconnect.

## 2026-10-23 — Harmonize + FFmpeg preset verification

- PC harmonize implemented (`cpu-harmonize`): mask color match + seam feather;
  no-mask mode does a mild auto-level.
- `pc/tools/test_ffmpeg_presets.py`: **23/23 passed** (10 video, 8 audio, 5 image).
- `pc/tools/test_pc_features.py` re-run: **20/20 passed**.
- WebUI default resolution changed to 512×512 to match the ONNX model.
- Note: MP3 192/320 are capped by a 16 kHz mono source (max ~160 kbps).
- Android harmonize implemented (`ImageOps.kt`, `engine=cpu-harmonize`);
  build `BUILD SUCCESSFUL`.

## 2026-10-23 — GPU/CPU, themes, language

- PC ONNX: provider auto-selection (CUDA → DirectML → CPU); CPU threads set to
  `cpu_count-1`. If `onnxruntime-gpu` is installed, CUDA + CPU split the graph.
- Android: acceleration setting Auto/CPU/GPU (NNAPI), persisted; engine reloads
  on change.
- PC WebUI: dark/light toggle (CSS variables, localStorage).
- Android: theme setting System/Light/Dark via `AppCompatDelegate`; color
  resources in `values` + `values-night`.
- Android: language setting English/Slovak/Czech/German via
  `AppCompatDelegate.setApplicationLocales`; strings extracted for tabs, titles,
  main buttons; base English, translations in `values-sk/cs/de`.
- Verified: Android `BUILD SUCCESSFUL`; PC generate 200; WebUI serves the theme
  toggle; PC ONNX providers log `['CPUExecutionProvider']` on the CPU-only
  onnxruntime build.

## 2026-10-23 — Android on-device test

- Installed the latest APK on Vivo V2415; UI shows Theme (System), Language
  (English), Acceleration (Auto); Inpaint tab present.
- Passed: health/models/catalog/settings, generate 256 (valid PNG) and 384
  (negative prompt + guidance), edits (img2img), inpaint color and add,
  harmonize (cpu-harmonize), transcription (whisper-large-v3-turbo).
- Expected 503: caption, upscale, FFmpeg.
- **Bug:** inpaint `add` succeeded, then `remove` lost the connection; the app
  process appeared to be recreated (`pidof` changed, engine re-init took ~109 s
  once). Root cause not yet proven: likely native memory pressure (ORT +
  XNNPACK) or the OEM process killer. Next step: capture
  `lowmemorykiller`/`am_kill` logs and native heap before/after each job, then
  test CPU-only acceleration and a delayed (not immediate) engine release.
- Device disconnected again after the test.

## 2026-10-23 — Audit fixes + on-device re-test

Fixed from the audit:
- **P0 Android inpaint composite** — now uses `alpha * luminance`; verified on
  device: diff inside mask 72.8, outside 0.0 (was 118.3 outside).
- **P0 PC img2img/inpaint schedule** — `_denoise` accepts the sliced timestep
  list; `img2img` and `inpaint` pass it. Verified 200 + valid outputs.
- **P0 Android non-square size** — engine is square; uses the larger side and
  reports `size` (actual) + `requested_size`; `n` now generates distinct images.
- **P1 PC ModelManager** — `ensure` switch calls `on_unload` after releasing the
  lock.
- **P1 Android ModelManager** — removed `System.gc()` from the lock; translated
  comments.
- **P2** — removed the leftover `assets/webui.html`; PC `quality=high` → 8 steps;
  `n` uses distinct seeds; WebUI inline colors use CSS variables; remaining
  Slovak comments translated (`ApiServer`, `AudioDecoder`, `WhisperLib`,
  `ModelManager`, `ffmpeg_tools`, `registry`, `vlm`, `ytdlp`, `unified_server`).

On-device re-test (Vivo V2415, PID stable 2390 across all jobs):
| Test | Result |
|---|---|
| generate 256×128, n=2 | 200, `size=256x256`, `requested_size=256x128`, 2 distinct images |
| inpaint add | 200, inside 72.8 / outside 0.0 |
| inpaint remove | 200, engine=onnx |
| inpaint color | 200, engine=onnx |
| harmonize | 200, engine=cpu-harmonize |
| transcribe | 200, whisper-large-v3-turbo |
| generate 256 after the above | 200, PID unchanged |

Repo cleanup: 109 scratch/test files removed (root untracked 128 → 29).

## 2026-10-23 — Verification closure (concurrency, states, load)

- **PC ModelManager race fixed:** `on_unload("switch")` now runs while the lock
  is held, so a concurrent `ensure` cannot install a new model before the old
  engine is released. `unload()` also calls the hook under the lock.
- `pc/tools/test_model_manager_concurrency.py`: PASS — basic switch, release-before-new-
  state (second ensure blocked ≥0.15 s during a 0.25 s callback), 8 threads ×
  10 switches with no deadlock and `unload calls == load_count - 1`.
- **Android state reporting:** Ktor `Monitoring` intercept tracks image/audio
  requests; `ModelManager.beginRequest/endRequest` updates `busy/queue/active`;
  `setLoading` wraps LCM init. On-device: `busy=true`, `loading=true` during
  init, then `loading=false`; after completion `busy=false, queue=0`.
- **Android mask polarity:** black 0.0, white 145.0, gray 62.1, alpha 62.1.
- **Android load test:** 20 mixed requests → 20/20 OK, PID 12393 unchanged,
  threads 55–56, RSS 1.9–3.9 GB (end 2.66 GB), no OOM/restart.
- **PC regressions:** high → 8 steps, n=2 distinct, size metadata, black/white/
  gray inpaint masks 0.0 / 86.4 / 35.1.
- Git: new docs/source/tests are untracked; commit plan proposed (not executed,
  pending approval).

## 2026-10-23 — English-only codebase

- Translated all Slovak comments, docstrings, UI strings, log messages and model
  notes to English across Android (Kotlin + XML), PC (Python + WebUI).
- Verified 0 Slovak diacritics in `android/**/*.kt`, `pc/unified/**/*.py`,
  `pc/unified_server.py`, `pc/download_models.py`, `pc/webui_unified.html`.
- Deleted unused `RemoteProxy.kt` (remote mode removed).
- Docs translated by a subagent (ANDROID, HANDOFF, UNIFIED-CONTRACT, UNIFIED-OPS).
- Device caveat: Vivo sometimes backgrounds the activity immediately
  (`CanvasContext destroyed`), which breaks synthetic taps in automation.
  Manual use is unaffected.

## 2026-10-23 — PC busy/queue truth

Audit asked whether PC `busy`/`queue` reflect reality. They did not:
`ModelManager.begin()/end()` only updated `in_flight`, while `_busy` and `_queue`
were initialized once and never touched.

- Fix: `_refresh_busy_locked()` sets `busy = in_flight > 0` and
  `queue = max(0, in_flight - 1)` after every `begin()`/`end()`. The HTTP
  middleware already wraps POST `/v1/images/*`, `/v1/audio/*`, `/v1/chat/*`
  in `begin()`/`end()` with `finally`, so failures reset the state too.
- `pc/tools/test_model_manager_concurrency.py` gained `test_begin_end_updates_busy_and_queue`
  (single, nested, drain, unbalanced end) — PASS.
- New `pc/tools/test_pc_states.py` against a live server — PASS:
  | Step | Observed |
  |---|---|
  | idle | `busy=false, queue=0, in_flight=0` |
  | 1× 512 high | `busy=true, in_flight=1, queue=0` |
  | 2 concurrent | `busy=true, queue=1, in_flight=2`, both 200 |
  | after | `busy=false, queue=0, in_flight=0` |
  | invalid inpaint (400) | state back to idle |
- `pc/tools/test_pc_features.py` now asserts the earlier fixes as regressions:
  `n=2` returns 2 distinct images, `quality=high` → `steps=8`, non-square
  request reports `size=320x320` + `requested_size=320x256`.

## 2026-10-23 — Android concurrent state test (partial)

New `android/tests/test_android_states.py` starts a 384×384 generation and an inpaint
concurrently against the running app.

- Verified before the disconnect: `busy=true, queue=1` while both were in
  flight — the queue counter does reflect a second request.
- The completion half could not be verified: USB/ADB lost the device mid-test
  (`adb devices` empty, no Android ADB device in Windows PnP), both clients got
  "server disconnected". No logcat, so the cause is unclassified — it may be a
  cable/USB drop or a process death. Re-run when the phone is attached and
  capture `adb logcat` around the test.
- Repo: removed `test_lcm_tmp.py` and `test_android_continue.py` (scratch /
  superseded); `pc/tools/test_pc_features.py` and `android/tests/test_android_features.py` now write
  sample outputs to a temp directory instead of the repo root.

## 2026-10-24 — PC upscale ONNX fallback, Android NCNN/VLM skeletons

**PC upscale:** `pc/unified/upscale.py` now tries Real-ESRGAN in this order:
torch (RRDBNet) → ONNX (`RealESRGAN_x4plus.onnx`) → Pillow LANCZOS.
Tiled inference (256×256, overlap 16) keeps VRAM low. Scale 2/3 downsample from
native x4. DoS cap 8192px. Verified import, no model weights present → Pillow fallback.

**Android upscale skeleton:** Added `RealEsrganNcnnEngine.kt` (NCNN Vulkan),
wired in `ApiServer.kt` POST `/v1/images/upscale`.
- Downloads `realesrgan-ncnn-vulkan` ZIP (param + bin) from GitHub.
- JNI bridge `librealesrgan_jni.so` placeholder (requires NDK build with NCNN).
- Returns 503 until native lib is built and bundled.

**Android VLM skeleton:** Added `SmolVlmEngine.kt` (llama.cpp mtmd),
wired in `ApiServer.kt` POST `/v1/images/caption` + `/v1/chat/completions`.
- Downloads GGUF model + mmproj from Hugging Face.
- JNI bridge `libsmolvlm_jni.so` placeholder (requires NDK build with mtmd).
- Returns 503 until native lib is built and bundled.

**Android API updates:**
- `switchTo()` releases NCNN/VLM engines when switching away.
- `appContext` used for engine init (fixes build error).
- Removed `companion object` from engine objects (not allowed in `object`).

**Checklist updated:** Upscale/VLM entries show skeleton wired, JNI placeholder.

**Build:** Android `assembleDebug` PASS. PC `pytest tests -q` 44 passed.

## 2026-10-23 — Size rule, language sweep, ffmpeg pass log

- **Size rule unified:** PC uses the larger requested side (`max(w, h)`,
  clamped to 256–512) like Android, in `_try_real_generation`, generations
  metadata, edits and inpaint. Verified: `320x256` → `size=320x320`,
  `requested_size=320x256`.
- `pc/tools/test_pc_features.py` regression section: `n=2` → 2 distinct images,
  `quality=high` → `steps=8`, size metadata. Full suite **29/29 passed**;
  `pc/tools/test_ffmpeg_presets.py` **23/23 passed**.
- **Language sweep:** the earlier "0 Slovak" check only looked at diacritics
  in Kotlin. `AndroidManifest.xml`, `cpp/CMakeLists.txt`,
  `cpp/whisper_jni.cpp`, `build.gradle.kts`, `ModelCatalog.kt`,
  `ModelDownloader.kt`, WebUI, `requirements.txt`, `ffmpeg_tools.py`,
  `lcm_onnx.py`, `tests/*` docstrings, the Android `.ps1` scripts,
  `android/tools/monitor_android.ps1`, `Switch-Service-ToUnified.ps1` and the archived plan
  were still Slovak; all translated. Re-scan: **0 hits**.
- **ffmpeg two-pass log leak fixed:** `-passlogfile` points next to the
  temporary output and `os.devnull` is used; `ffmpeg2pass-0.log` no longer
  lands in the server working directory (re-verified after deletion).
- `python -m pytest tests -q`: **44 passed**; Android `assembleDebug`:
  **BUILD SUCCESSFUL**.

## 2026-10-24 — GPU (CUDA) on PC

Hardware: RTX 3060 Laptop 6 GB, driver 617.42, CUDA 13.4, cuDNN 9.27
(`nvidia-cudnn-cu13` pip), `onnxruntime-gpu 1.31.0`.

Provider logic (`pc/unified/onnx_env.py`):
- `PRISM_ONNX_PROVIDER=auto` (default): CUDA → CPU. DirectML is skipped for
  `qdq` models because DML fails in `DmlGraphFusionHelper` on the INT8 graph.
- `cuda` / `dml` / `cpu` force a path.
- Windows: the pip `nvidia/*/bin` DLL directories are registered before the
  first session (`register_nvidia_dlls`).
- CUDA options: `cudnn_conv_algo_search=HEURISTIC`,
  `cudnn_conv_use_max_workspace=0`, `arena_extend_strategy=kSameAsRequested`,
  and `enable_mem_pattern=False` for GPU sessions.

Measured on the server (warm, LCM-Dreamshaper INT8):

| Request | CPU | CUDA |
|---|---|---|
| 256×256 / 4 steps | ~23 s | **6.3 s** |
| 384×384 / 6 steps | ~45 s | **10.7 s** |
| 512×512 / 6 steps | ~65 s | **51.8 s** |

Caveats (measured):
- The first run of each new shape pays cuDNN planning: 256 ≈ 43 s, 384 ≈ 65 s,
  512 ≈ 184 s (one-time per process; the first also includes engine init).
- VRAM sits at ~5.9 GB of 6 GB. `PRISM_ONNX_PROVIDER=cpu` is the safe fallback.
- ORT reports 777 Memcpy nodes for CUDA on this QDQ graph ("unable to run CUDA
  graph"), so the gain is modest at 512.
- `release_pipeline()` frees the ONNX sessions on idle/model switch, so the GPU
  is not held forever.
- Found and fixed `/v1/audio/translations` 500: it called `transcriptions()`
  without `url`, so the default was a FastAPI `Form` marker. The suite now
  covers translations (**31/31 passed**).
- GPU output validity: 256/384/512 PNG dimensions OK; inpaint mask polarity
  black 0.0 / white 86.1 / gray 35.1; img2img 256×256. Peak GPU utilisation
  53% during a 512 request (VRAM 5.7 GB), state returns to idle afterwards.

AMD: the machine also has a Radeon iGPU. DirectML cannot execute this QDQ
graph (`The parameter is incorrect` in DmlGraphFusionHelper) and the iGPU
shares system memory. AMD support therefore needs `onnxruntime-directml` in a
separate env with a non-QDQ (FP16/FP32) model; with the current INT8 asset it
stays on CPU.

## 2026-10-25 — Android ORT crash fixed

Symptom: two concurrent image requests killed the process with SIGSEGV; the
20-request load test also died after ~18 requests.

Tombstone (logcat, `debuggerd`):
```
Fatal signal 11 (SIGSEGV), code 1 (SEGV_MAPERR) in tid eventLoopGroupP
#00..#09 libonnxruntime.so
#10 Java_ai_onnxruntime_OrtSession_run
#29 LcmOnnxEngine.runUnet
#34 LcmOnnxEngine.denoise
#44 LcmOnnxEngine.generate
```
Root causes:
1. Heavy ONNX inference ran on the Ktor/Netty event-loop thread.
2. Two ORT `run()` calls could overlap (no serialization).
3. The engine was released after every request and re-initialised on the next
   one; the repeated session churn added native fragmentation.

Fix (`ApiServer.kt`):
- `heavyDispatcher = Dispatchers.IO.limitedParallelism(1)`; every heavy path
  (init + generate / img2img / inpaint / upscale / caption) runs in
  `withContext(heavyDispatcher)`, so requests queue without blocking the
  event loop and only one ORT run is active.
- The LCM engine stays **resident** between requests; a coroutine releases it
  after 10 minutes without image work (idle unload). Switching to upscale or
  VLM also releases it to free RAM.
- Per-request `LcmOnnxEngine.release()` removed (it caused the churn).

Verified on the device (Vivo V2415, Android 16):
| Test | Result |
|---|---|
| concurrent state | `queue=1`, both 200, PID stable, state returns to idle |
| feature suite | **21/21** |
| mask polarity | black 0.0 / white 144.6 / gray 62.0 / alpha 62.0 |
| load test (20 mixed) | **20/20**, PID 13108 stable, RSS 3.69–3.78 GB, threads 96 |
| warm generate 256/4 | **7–10 s** (was 25–33 s with re-init) |
| inpaint add/remove | **~9 s** (was ~25 s) |
| inpaint color | 0.2 s |
| crash in logcat | none |

## 2026-10-25 — Upscale and VLM fixes

Two old gaps closed, plus two real bugs found in the torch upscale path.

**Real-ESRGAN architecture (bug):** the vendored RRDBNet did not match
BasicSR. `RRDB` summed three dense blocks in parallel and the network was
missing the LeakyReLU after `conv_up1`, `conv_up2` and `conv_hr`. On a 0.5
gray input the old model emitted ~2.2e10; the fixed model emits 0.508.
Consequence: the torch upscale path had always produced garbage.

**ONNX upscale (new):** `pc/tools/convert_esrgan_onnx.py` loads the `.pth` with the
correct key names (`conv1..conv5`, `conv_up1/2`), exports with dynamic H/W,
inlines the external weights and validates against torch:
`max abs diff 2e-6`, single 67 MB `RealESRGAN_x4plus.onnx`.
`find_onnx_weight` accepts inline or `.onnx.data` weights.
`PRISM_UPSCALE_ENGINE=auto|torch|onnx|pillow` selects the path.

**Single-resident fix:** upscale now calls
`_manager.ensure("upscale", "RealESRGAN_x4v3")`, and `_release_engines` calls
`release_upscale()`. Switching from generation to upscale frees the 5.9 GB
LCM sessions, so ESRGAN can use the GPU instead of spilling to CPU.
Measured 512→1024: **torch CUDA 5.4–6.2 s** (was 117 s via CPU ONNX),
ONNX CUDA 13.6 s, ONNX CPU 27 s on a 256×256 crop.

**VLM (new):** `torch 2.14.1+cu130` + `transformers 5.20.2` installed; the
`models/SmolVLM_500M` snapshot was already present. Caption and vision chat
return real text now: caption first call 42 s (load included), cached ~7 s;
chat ~3 s. The manager still releases the LCM pipeline before VLM and the VLM
before generation, so only one heavy engine is resident.

**Verification:** `pc/tools/test_pc_features.py` **37/37 passed** (adds upscale,
caption and chat with assertions), ffmpeg 23/23, states PASS, `pytest` 44.
GPU upscale engine reported as `realesrgan-torch`.

**AMD:** unchanged and documented — DirectML cannot execute the INT8 QDQ
graph, and the iGPU shares system memory. A non-QDQ FP16/FP32 export would be
needed; the discrete NVIDIA GPU is the practical path on this machine.

## 2026-10-26 — Repository split: pc/ and android/

The repository now contains two self-contained products:

```
pc/        PC server, unified package, WebUI, models, tests, tools,
           scripts, PC docs (models/ + models.json live here)
android/   Gradle app, native engines, tests, tools, Android docs
docs/      shared project docs
testdata/  shared test fixtures
```

- Root README explains that a user only needs `pc/` or `android/`.
- PC e2e drivers moved to `pc/tools`, pytest suite to `pc/tests`; helper
  scripts to `pc/scripts`, legacy image-only launcher to `pc/legacy`.
- Android e2e drivers moved to `android/tests`, helper scripts to
  `android/tools`, doc to `android/docs`.
- Paths in tests/tools/scripts/docs updated; `pc/pytest.ini` + `conftest.py`
  added so the suite runs from `pc/`.
- Verified after the move: `pytest` **44 passed** from `pc/`, e2e
  **37/37** against the server started from `pc/`, Android
  `assembleDebug` **BUILD SUCCESSFUL**.

**Windows service:** `OdysseusDiffusion` still points at the old root paths.
Run `pc/scripts/migrate-service-paths.ps1` as Administrator once to repoint
the SCM binary to `pc/scripts/nssm.exe` and the app parameters to
`pc/diffusion_server.py` + `pc/models.json`. A temporary root `nssm.exe`
(gitignored) keeps the SCM path valid until then; the service will fail on
restart before the migration because its script path no longer exists.

## Prism PC launcher and transcription model (2026-10-26)

**Models contract.** `/v1/models` is the same contract for both products, so it
now carries `kind` and `installed`. The installable catalog
(`models_manager.catalog()`), not `build_registry`'s `missing` flag, decides
both: image entries come from the registry, `upscale`/`vlm`/`audio` from the
catalog, and an audio size counts as installed when
`pc/models/faster-whisper-<name>/model.bin` exists and is larger than 50 MB.
The WebUI therefore no longer hardcodes model lists.

**Whisper device.** CTranslate2 4.x needs the CUDA 12 runtime
(`cublas64_12.dll`, `cudnn64_9.dll`); this PC only has CUDA 13, so a
CUDA-attached whisper model fails in the first `encode()`. Transcription
defaults to CPU int8 (~4.5x realtime: 539 s for 40:57), which is sufficient,
and `WHISPER_DEVICE=cuda` opts into the GPU when the CUDA 12 DLLs exist. A
CUDA failure falls back to CPU inside the same request instead of failing it.

**No splitting for the local engine.** faster-whisper already does VAD and
30 s windowing; the old splitter used 25 s chunks with 5 s overlap and reloaded
the model for every chunk. Long files now go through in one pass; splitting
stays for the remote `WhisperApi` path.

**Evidence.** `pytest` 44 passed from `pc/`; live server on 8102 returns
`kind` for audio/image/upscale/vlm with `small installed=True`; the WebUI
renders as Prism with 9 tabs, populated dropdowns and the 500 MB label;
`POST /v1/audio/transcriptions` on `testdata/audio/jfk.wav` -> 200 with the
exact JFK sentence; the 40:57 recording produced 272 segments in 539 s.

**Port.** 8101 is still held by the legacy `OdysseusDiffusion` service, so the
launcher detects an occupied port and starts Prism on 8102. The permanent fix
is the Administrator migration script.

**Whisper residency truth (2026-10-26).** `_WHISPER_CACHE` holds the
faster-whisper weights, while `ModelManager` only tracks the `audio` kind, so
`/health` used to report `audio_loaded=false` with a model in memory. It now
reports `audio_model`, `audio_loaded` and `audio_whisper_models_resident` from
that cache. Verified: `False/0` -> transcription -> `True/1` -> image
generation -> `False/0`, i.e. `_release_engines` drops whisper on a model
switch. `pytest` 44 passed after the change.

## Prism WebUI dropdown root cause (2026-10-26)

**Cause:** the stylesheet declared `--bg: var(--bg)` (and the same for every
other token) in `:root`, i.e. each custom property referenced itself. Invalid
at computed-value time, so `background: var(--panel)` and `color: var(--text)`
resolved to nothing: the page fell back to a white background while inputs
still hardcoded `color: #fff`. That is why the selects looked empty and
oversized. Fixed by declaring the real dark palette and by using
`var(--text)` in inputs/buttons.

**Cause of "image models under Transcribe":** the single `#model` select sat
outside the per-tab forms, so it was visible on every tab. It now lives in
`#imagemodelwrap`, shown only for the image tabs.

**Cause of duplicate options:** `/v1/models` appended the whisper sizes twice
(once from `image_registry()["audio_models"]`, once from the installable
catalog, which now also contains them). The registry loop owns the audio list
and the catalog loop only adds ids that are not listed yet.

**Downloads.** A model without a verified source gets no button. Only repos
that answered HTTP 200 with a known `content-length` are in `WHISPER_SOURCES`,
each file with its exact byte size; `Systran/faster-whisper-large-v3-turbo`
answers 401, so `turbo` uses `deepdml/faster-whisper-large-v3-turbo-ct2` with a
pinned sha256. Installed state is derived from the files (exact `model.bin`
size, parseable `config.json`, existing `tokenizer.json`), never from a flag,
so an aborted download cannot look complete. Verified with a fresh 77 MB
`tiny` install (10 s) and an uninstall.

## PC server end-to-end test (2026-10-26)

**Why the suite asserts content, not status codes.** The first version checked
`r.status_code == 200` and passed while the server was generating 320x320 for a
320x256 request. Reading the produced PNG and comparing pixels is what exposed
the three bugs below.

**Registry lifetime.** `diffusion_server._init_lazy()` only ran in that
module's own FastAPI lifespan. Because `unified_server` imports the module
instead of running its app, `_REGISTRY` was empty and `ensure_model_loaded()`
answered 400 "Unknown model"; `_try_real_generation` caught it, logged a
warning and fell through to the ONNX LCM pipeline. The response had no engine
field, so nothing told the client. `ensure_registry()` now initialises the
registry on first use and the response carries `engine`.

**Path portability.** Absolute paths in `models.json` broke with the `pc/`
split. Relative paths resolved against the registry file make the product
folder self-contained, matching the "download only `pc/`" requirement.

**Outpaint.** Implemented on top of the existing inpaint pipeline: the image is
edge-replicated to the target canvas, the border becomes the mask, strength is
0.99, and the result is composited back so the original pixels stay exact
(measured centre diff 0.70 on a 512->768 run). Caps are 2048 px per side and
4 MP, returning 400 with the computed size otherwise.

**Evidence.** 39/39 checks, including pixel-level assertions and ffprobe
metadata; artifacts in `%TEMP%\prism_full_test` (`gen_512.png`, `inpaint_add.png`,
`outpaint_768.png`, `upscale_1024.png`, `results.json`).

**img2img engine order.** `/v1/images/edits` used to try the ONNX LCM engine
first, which silently capped every edit at 512 square. It now calls
`diffusion_server._run_edit` under the model lock (the model the user picked,
the size they asked for), then ONNX, then text2img, then a placeholder, and the
`engine` field says which one produced the pixels. Verified with a 640x384 edit
on DreamShaper_8: `engine=diffusers`, output 640x384, mean|diff| 17.8 against the
source.
