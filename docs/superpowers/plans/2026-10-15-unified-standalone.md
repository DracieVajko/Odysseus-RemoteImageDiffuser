# Unified Standalone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge DiffusionServer (:8101) + WhisperApi (:4444) into one lazy-loading backend with the same Odysseus contract + upscale + ffmpeg, plus a Kotlin Android scaffold as a local server.

**Architecture:** One FastAPI process `unified_server.py` with `images` + `audio` + `system` routers, a shared `ModelManager` (1 heavy model in RAM/VRAM, RLock, 503 loading/will-not-load). Android: Kotlin ForegroundService + Ktor server, same routes, inference via whisper.cpp JNI + LCM-Dreamshaper INT8 NCNN/Vulkan + Real-ESRGAN NCNN + ffmpeg-kit Min (LGPL).

**Tech Stack:** Python 3.12 FastAPI, diffusers 0.40, faster-whisper/CTranslate2, Pillow, ffmpeg binary, pytest. Android: Kotlin, Ktor, NCNN Vulkan, whisper.cpp JNI, ffmpeg-kit Min.

**Spec:** Approved in the 2026-10-15 conversation — Goals/Non-goals: V1 = gen+edit/inpaint+upscale+transcription+ffmpeg, API frozen, VLM only in V2, mobile primarily local (12GB) with a Remote fallback, steps cap 4-8 on mobile, Vulkan not NNAPI.

## Global Constraints

- The API must stay compatible: GET /health, GET /v1/models, POST /v1/images/generations|edits, POST /v1/images/inpaint|harmonize (JSON b64), POST /v1/audio/transcriptions|translations, GET /v1/images/progress/{id}, GET /webui.
- New: POST /v1/images/upscale {image b64, scale, model} — Real-ESRGAN-x4v3 default, tile processing.
- Always at most 1 heavy model resident; switching = unload + load; 503 loading (retry_after 30) and 503 will-not-load without hanging.
- ffmpeg: audio→wav 16kHz mono, video→audio extract, 100 MB limit, allowed extensions as in WhisperApi.
- Android: NCNN Vulkan (not NNAPI), steps 4-8 cap, lazy model download (not in the APK), ffmpeg-kit Min/LGPL only (not Full/GPL).
- Research verdicts are binding: LCM-Dreamshaper INT8 is the main mobile candidate, SD-Turbo has a non-commercial license, SSD-1B/XL and PaliGemma-3B are PC-only, whisper small Q5_0 is the minimum for SK, SmolVLM/Moondream2 only in V2.

## Review Focus

- Switching SD→Whisper at runtime does not free VRAM and the OOM killer kills the process — expect a clean unload + empty_cache.
- A huge canvas (8192px) must not hang — expect 503 will-not-load or CPU offload.
- An empty / bad audio file returns 400/415, not 500.
- Concurrent generations during a cold load return 503 loading, not a corrupt pipe.
- Android OOM: SD (2.5GB) + Whisper small (350MB) + upscale at once must not exceed ~3-4GB — expect a sequential ModelManager on mobile too.

---

### Task 1: Contract tests (TDD RED baseline)

**Files:**
- Create: `tests/test_unified_contract.py`
- Modify: no production files

**Interfaces:**
- Consumes: docs/ODYSSEUS-CONTRACT.md + WhisperApi README API
- Produces: the list of paths that Tasks 2-4 must implement

- [ ] **Step 1: Write the failing test**
```python
def test_unified_contract_paths_exist():
    # GET /health, GET /v1/models, POST /v1/images/generations|edits|inpaint|harmonize|upscale, POST /v1/audio/transcriptions|translations, GET /v1/images/progress/{id}
    assert unified_app_has_all_paths()
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_unified_contract.py -v`
Expected: FAIL (unified_server does not exist)
- [ ] **Step 3: Commit test only**
```bash
git add tests/test_unified_contract.py
git commit -m "test: unified contract paths"
```

### Task 2: ModelManager (shared VRAM/RAM gate)

**Files:**
- Create: `unified/__init__.py`, `unified/model_manager.py`
- Test: `tests/test_model_manager.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ModelManager.ensure(kind: str, id: str, pixels: int) -> str`, `ModelManager.unload(reason) -> bool`, `ModelManager.status() -> dict {loaded_kind, loaded_id, loading, busy, queue}`, `ModelBusyError`, `ModelLoadingError(retry_after)`

- [ ] **Step 1: Write the failing test**
```python
def test_only_one_resident():
    mm.ensure("image", "DreamShaper_8")
    mm.ensure("audio", "small")
    assert mm.status()["loaded_kind"] == "audio"  # the image model was unloaded
def test_loading_503_shape():
    with pytest.raises(ModelLoadingError) as e: mm.ensure_busy_case()
    assert e.value.retry_after == 30
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_model_manager.py -v`
Expected: FAIL
- [ ] **Step 3: Implement `ModelManager` in `unified/model_manager.py` (RLock, 1 resident, loading flag, busy/queue counters)**
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_model_manager.py tests/test_unified_contract.py -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add unified/model_manager.py tests/test_model_manager.py
git commit -m "feat: shared ModelManager single-resident gate"
```

### Task 3: ffmpeg_pre (audio/video normalization)

**Files:**
- Create: `unified/ffmpeg_pre.py`
- Test: `tests/test_ffmpeg_pre.py`

**Interfaces:**
- Consumes: nothing
- Produces: `normalize_audio_to_wav16k(src: Path, dst: Path) -> dict`, `ALLOWED_AUDIO_EXT`, `MAX_FILE_SIZE_MB=100`

- [ ] **Step 1: Write the failing test**
```python
def test_rejects_unsupported_ext(tmp_path):
    with pytest.raises(UnsupportedFormat): normalize_audio_to_wav16k(tmp_path/"x.xyz", tmp_path/"o.wav")
def test_rejects_empty_file(tmp_path):
    (tmp_path/"e.wav").write_bytes(b"")
    with pytest.raises(EmptyFile): normalize_audio_to_wav16k(tmp_path/"e.wav", tmp_path/"o.wav")
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_ffmpeg_pre.py -v`
Expected: FAIL
- [ ] **Step 3: Implement `unified/ffmpeg_pre.py` (ffprobe/ffmpeg when available, otherwise passthrough wav, validate ext/size/empty)**
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ffmpeg_pre.py -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add unified/ffmpeg_pre.py tests/test_ffmpeg_pre.py
git commit -m "feat: ffmpeg audio normalize gate"
```

### Task 4: unified_server.py (join without duplicating logic)

**Files:**
- Create: `unified_server.py`, `unified/registry.py`, `registry.json`
- Modify: `models.json` (do not touch the values, read only), `webui.html` (add the upscale + audio tab in a later commit)
- Test: `tests/test_unified_contract.py` (extend with upscale + audio)

**Interfaces:**
- Consumes: ModelManager.ensure/status, ffmpeg_pre.normalize, the existing diffusion_server ensure_model_loaded + Whisper _ensure_model_locked (called through a thin layer, not copied)
- Produces: FastAPI `app` with the paths from Task 1 + `GET /` → redirect /webui

- [ ] **Step 1: Extend the contract test with upscale + audio shapes**
```python
def test_upscale_contract(): assert upscale_response_has_image_and_elapsed()
def test_audio_transcription_contract(): assert transcription_has_text_field()
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_unified_contract.py -v`
Expected: FAIL
- [ ] **Step 3: Implement `unified_server.py` (import diffusion_server + faster-whisper lazy, routers delegate, ModelManager gate before every inference, 503 shapes from ODYSSEUS-CONTRACT, legacy :4444 shim note in the README)**
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_unified_contract.py tests/test_model_manager.py tests/test_ffmpeg_pre.py -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add unified_server.py unified/registry.py registry.json tests/test_unified_contract.py
git commit -m "feat: unified server image+audio same-contract"
```

### Task 5: Upscale endpoint (Real-ESRGAN-x4v3 default, Pillow fallback)

**Files:**
- Create: `unified/upscale.py`
- Modify: `unified_server.py` (wire POST /v1/images/upscale)
- Test: `tests/test_upscale.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `upscale_image_b64(image_b64: str, scale: int) -> {image_b64, elapsed}` (Pillow LANCZOS fallback until the Real-ESRGAN weights are present)

- [ ] **Step 1: Write the failing test**
```python
def test_upscale_2x_grows_canvas():
    out = upscale_image_b64-tiny-red-dot-2x()
    assert out_width == 2*in_width
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_upscale.py -v`
Expected: FAIL
- [ ] **Step 3: Implement `unified/upscale.py` (Pillow + tile hook ready for NCNN/Real-ESRGAN, DoS cap 8192px → 503)**
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_upscale.py -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add unified/upscale.py tests/test_upscale.py
git commit -m "feat: upscale endpoint with tile hook"
```

### Task 6: Android scaffold (Kotlin, Ktor, same contract)

**Files:**
- Create: `android/README.md`, `android/settings.gradle.kts`, `android/app/build.gradle.kts`, `android/app/src/main/.../MainActivity.kt`, `android/app/src/main/.../ApiServer.kt`, `android/app/src/main/.../ModelManager.kt`, `android/contract-compat.md`
- Test: `android/contract-compat.md` checklist (no emulator: curl shapes)

**Interfaces:**
- Consumes: ODYSSEUS-CONTRACT + research table
- Produces: Ktor paths /health /v1/models /v1/images/* /v1/audio/* with 503 shapes, Local/Remote/Auto switch, steps cap 4-8, Vulkan note, ffmpeg-kit Min dependency

- [ ] **Step 1: Write contract-compat.md with the expected curl examples**
- [ ] **Step 2: Create the Gradle + Ktor skeleton (no heavy .bin files, models via a download-manager stub)**
- [ ] **Step 3: Verify the `./gradlew :app:assembleDebug` instruction in the README (build only on a machine with the Android SDK)**
- [ ] **Step 4: Commit**
```bash
git add android/
git commit -m "feat: android ktor scaffold same-contract"
```

### Task 7: WebUI unify + docs (OPERATIONS, INSTALL)

**Files:**
- Modify: `webui.html` → `webui_unified.html` (add the Generate/Edit-Upscale/Transcribe tabs, mask editor canvas, progress poll)
- Create: `docs/UNIFIED-CONTRACT.md`, `docs/ANDROID.md`
- Test: manual `http://127.0.0.1:8101/webui` checklist

- [ ] **Step 1-5: TDD for docs links + webui tabs existing**
Run: `pytest tests/ -k "not slow" -q`
Expected: PASS
Commit: `docs: unified webui + android wiring`
