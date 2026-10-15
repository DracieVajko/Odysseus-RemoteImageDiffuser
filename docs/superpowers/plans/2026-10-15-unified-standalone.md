# Unified Standalone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zlúčiť DiffusionServer (:8101) + WhisperApi (:4444) do jedného lazy-loading backendu s rovnakou Odysseus zmluvou + upscale + ffmpeg, plus Kotlin Android scaffold ako lokálny server.

**Architecture:** Jeden FastAPI proces `unified_server.py` s routermi `images` + `audio` + `system`, zdieľaný `ModelManager` (1 ťažký model v RAM/VRAM, RLock, 503 loading/will-not-load). Android: Kotlin ForegroundService + Ktor server, rovnaké cesty, inferencia whisper.cpp JNI + LCM-Dreamshaper INT8 NCNN/Vulkan + Real-ESRGAN NCNN + ffmpeg-kit Min (LGPL).

**Tech Stack:** Python 3.12 FastAPI, diffusers 0.40, faster-whisper/CTranslate2, Pillow, ffmpeg binary, pytest. Android: Kotlin, Ktor, NCNN Vulkan, whisper.cpp JNI, ffmpeg-kit Min.

**Spec:** Schválené v konverzácii 2026-10-15 — Goals/Non-goals: V1 = gen+edit/inpaint+upscale+transkripcia+ffmpeg, API frozen, VLM až V2, mobil primárne lokálny (12GB) s Remote fallbackom, steps cap 4-8 na mobile, Vulkan nie NNAPI.

## Global Constraints

- API musí zostať kompatibilná: GET /health, GET /v1/models, POST /v1/images/generations|edits, POST /v1/images/inpaint|harmonize (JSON b64), POST /v1/audio/transcriptions|translations, GET /v1/images/progress/{id}, GET /webui.
- Nové: POST /v1/images/upscale {image b64, scale, model} — Real-ESRGAN-x4v3 default, tile processing.
- Vždy max 1 ťažký model rezidentný; prepnutie = unload + load; 503 loading (retry_after 30) a 503 will-not-load bez hangovania.
- ffmpeg: audio→wav 16kHz mono, video→audio extrakt, limit 100 MB, povolené ext ako vo WhisperApi.
- Android: NCNN Vulkan (nie NNAPI), steps 4-8 cap, modely lazy download (nie v APK), ffmpeg-kit Min/LGPL only (nie Full/GPL).
- Research verdikty záväzné: LCM-Dreamshaper INT8 hlavný kandidát mobile, SD-Turbo nekomerčná licencia, SSD-1B/XL a PaliGemma-3B PC-only, whisper small Q5_0 minimum pre SK, SmolVLM/Moondream2 až V2.

## Review Focus

- Prepnutie SD→Whisper za behu neuvoľní VRAM a OOM killer zabije proces — očakávame čistý unload + empty_cache.
- Obrovský canvas (8192px) nesmie hangnúť — očakávame 503 will-not-load alebo CPU offload.
- Prázdny / zlý audio súbor vráti 400/415, nie 500.
- Concurrent generations počas cold load vrátia 503 loading, nie corrupt pipe.
- Android OOM: SD (2.5GB) + Whisper small (350MB) + upscale naraz nesmie prekročiť ~3-4GB — očakávame sekvenčný ModelManager aj na mobile.

---

### Task 1: Contract tests (TDD RED základ)

**Files:**
- Create: `tests/test_unified_contract.py`
- Modify: žiadne produkčné

**Interfaces:**
- Consumes: docs/ODYSSEUS-CONTRACT.md + WhisperApi README API
- Produces: zoznam ciest ktoré Task 2-4 musia implementovať

- [ ] **Step 1: Write the failing test**
```python
def test_unified_contract_paths_exist():
    # GET /health, GET /v1/models, POST /v1/images/generations|edits|inpaint|harmonize|upscale, POST /v1/audio/transcriptions|translations, GET /v1/images/progress/{id}
    assert unified_app_has_all_paths()
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_unified_contract.py -v`
Expected: FAIL (unified_server neexistuje)
- [ ] **Step 3: Commit test only**
```bash
git add tests/test_unified_contract.py
git commit -m "test: unified contract paths"
```

### Task 2: ModelManager (zdieľaný VRAM/RAM gate)

**Files:**
- Create: `unified/__init__.py`, `unified/model_manager.py`
- Test: `tests/test_model_manager.py`

**Interfaces:**
- Consumes: nič
- Produces: `ModelManager.ensure(kind: str, id: str, pixels: int) -> str`, `ModelManager.unload(reason) -> bool`, `ModelManager.status() -> dict {loaded_kind, loaded_id, loading, busy, queue}`, `ModelBusyError`, `ModelLoadingError(retry_after)`

- [ ] **Step 1: Write the failing test**
```python
def test_only_one_resident():
    mm.ensure("image", "DreamShaper_8")
    mm.ensure("audio", "small")
    assert mm.status()["loaded_kind"] == "audio"  # image bol unloadnutý
def test_loading_503_shape():
    with pytest.raises(ModelLoadingError) as e: mm.ensure_busy_case()
    assert e.value.retry_after == 30
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_model_manager.py -v`
Expected: FAIL
- [ ] **Step 3: Implement `ModelManager` v `unified/model_manager.py` (RLock, 1 rezident, loading flag, busy/queue počítadlo)**
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_model_manager.py tests/test_unified_contract.py -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add unified/model_manager.py tests/test_model_manager.py
git commit -m "feat: shared ModelManager single-resident gate"
```

### Task 3: ffmpeg_pre (audio/video normalizácia)

**Files:**
- Create: `unified/ffmpeg_pre.py`
- Test: `tests/test_ffmpeg_pre.py`

**Interfaces:**
- Consumes: nič
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
- [ ] **Step 3: Implement `unified/ffmpeg_pre.py` (ffprobe/ffmpeg ak je, inak passthrough wav, validácia ext/size/empty)**
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_ffmpeg_pre.py -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add unified/ffmpeg_pre.py tests/test_ffmpeg_pre.py
git commit -m "feat: ffmpeg audio normalize gate"
```

### Task 4: unified_server.py (spojenie bez duplikácie logiky)

**Files:**
- Create: `unified_server.py`, `unified/registry.py`, `registry.json`
- Modify: `models.json` (nedotýkať hodnoty, len čítať), `webui.html` (pridať upscale + audio tab v ďalšom commite)
- Test: `tests/test_unified_contract.py` (rozšíriť o upscale + audio)

**Interfaces:**
- Consumes: ModelManager.ensure/status, ffmpeg_pre.normalize, existujúce diffusion_server ensure_model_loaded + Whisper _ensure_model_locked (volané cez tenkú vrstvu, nie kopírované)
- Produces: FastAPI `app` s cestami z Task 1 + `GET /` → redirect /webui

- [ ] **Step 1: Rozšíriť contract test o upscale + audio tvary**
```python
def test_upscale_contract(): assert upscale_response_has_image_and_elapsed()
def test_audio_transcription_contract(): assert transcription_has_text_field()
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_unified_contract.py -v`
Expected: FAIL
- [ ] **Step 3: Implement `unified_server.py` (import diffusion_server + faster-whisper lazy, routre delegujú, ModelManager gate pred každou inferenciou, 503 tvary z ODYSSEUS-CONTRACT, legacy :4444 shim poznámka v README)**
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
- Modify: `unified_server.py` (napojiť POST /v1/images/upscale)
- Test: `tests/test_upscale.py`

**Interfaces:**
- Consumes: nič nové
- Produces: `upscale_image_b64(image_b64: str, scale: int) -> {image_b64, elapsed}` (Pillow LANCZOS fallback kým nie je Real-ESRGAN váha)

- [ ] **Step 1: Write the failing test**
```python
def test_upscale_2x_grows_canvas():
    out = upscale_image_b64-tiny-red-dot-2x()
    assert out_width == 2*in_width
```
- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_upscale.py -v`
Expected: FAIL
- [ ] **Step 3: Implement `unified/upscale.py` (Pillow + tile hook pripravený pre NCNN/Real-ESRGAN, DoS cap 8192px → 503)**
- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_upscale.py -v`
Expected: PASS
- [ ] **Step 5: Commit**
```bash
git add unified/upscale.py tests/test_upscale.py
git commit -m "feat: upscale endpoint with tile hook"
```

### Task 6: Android scaffold (Kotlin, Ktor, rovnaká zmluva)

**Files:**
- Create: `android/README.md`, `android/settings.gradle.kts`, `android/app/build.gradle.kts`, `android/app/src/main/.../MainActivity.kt`, `android/app/src/main/.../ApiServer.kt`, `android/app/src/main/.../ModelManager.kt`, `android/contract-compat.md`
- Test: `android/contract-compat.md` checklist (bez emulatora: curl tvary)

**Interfaces:**
- Consumes: ODYSSEUS-CONTRACT + research tabuľka
- Produces: Ktor cesty /health /v1/models /v1/images/* /v1/audio/* s 503 tvarmi, Local/Remote/Auto prepínač, steps cap 4-8, Vulkan poznámka, ffmpeg-kit Min závislosť

- [ ] **Step 1: Napísať contract-compat.md s očakávanými curl príkladmi**
- [ ] **Step 2: Vytvoriť Gradle + Ktor skeleton (bez ťažkých .bin, modely cez download-manager stub)**
- [ ] **Step 3: Overiť `./gradlew :app:assembleDebug` inštrukciu v README (build až na stroji s Android SDK)**
- [ ] **Step 4: Commit**
```bash
git add android/
git commit -m "feat: android ktor scaffold same-contract"
```

### Task 7: WebUI unify + docs (OPERATIONS, INSTALL)

**Files:**
- Modify: `webui.html` → `webui_unified.html` (pridať tabs Generate/Edit-Upscale/Transcribe, mask editor canvas, progress-poll)
- Create: `docs/UNIFIED-CONTRACT.md`, `docs/ANDROID.md`
- Test: manuálny `http://127.0.0.1:8101/webui` checklist

- [ ] **Step 1-5: TDD pre docs linky + webui tabs existujú**
Run: `pytest tests/ -k "not slow" -q`
Expected: PASS
Commit: `docs: unified webui + android wiring`
