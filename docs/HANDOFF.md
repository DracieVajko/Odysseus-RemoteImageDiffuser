# HANDOFF — status as of 2026-10-23

> **Historical snapshot, kept for context only.** Current state lives in
> `docs/CHECKLIST.md` (what is done) and `docs/CHECKPOINTS.md` (why and
> with what evidence). Do not append to this file.

Quick context for continuing after a restart. The source of truth is the code, not this file.

## Done

### Android (Prism)
- `ModelDownloader.kt` — custom download engine (resume, staging, manifest, cancel, progress).
- `ModelCatalog.kt` — whisper-large-v3-turbo, LCM-Dreamshaper ONNX (QDQ), Real-ESRGAN NCNN, SmolVLM/SmolVLM2.
- `LcmOnnxEngine.kt` — full LCM pipeline (CLIP BPE tokenizer, text encoder, UNet, LCM scheduler, VAE decode).
  - The INT8 UNet was converted from QOperator (`ConvInteger`) to **QDQ** (`pc/tools/convert_lcm_qdq.py`); the phone runs the inline model (824 MB).
  - Text encoder `input_ids` must be **int32**.
- `HardwareProfile` — adaptive threads (1-8), image size according to RAM.
- Auto-split of long recordings (30s chunks) — verified on 2-minute and 45-minute files.
- UI: tab bar (Server / Generate / Transcribe / Upscale / Caption / Tools), swipe + slide/fade animation, active tab highlight, collapsible model sections, console (scrollable/selectable/copy), Save image, Save transcript, Stop download.
- **No WebUI, no Local/Auto/Remote, no remote host** — everything runs locally and offline.
- `largeHeap="true"` in the manifest (OOM fix on 45-minute files).
- English-only codebase.

### PC (pc/unified/)
- Fully local server (FastAPI + diffusers + faster-whisper + ffmpeg).
- Model management endpoints + custom URL download + real cancel.
- LCM ONNX pipeline without torch: `pc/unified/lcm_onnx_numpy.py`.
- WebUI: console (scrollable/copy), Save transcript, Stop download.

### Verified
- Transcription: 11s JFK → 200 (~27-32s with large-v3-turbo).
- 45-min SK file: chunking works (91 chunks), no crash.
- Image generation on Android: 256×256/4 ≈ 20-30s, 512×512/6 ≈ 60-70s, valid PNGs.
- Image generation on PC (no torch): ONNX fallback returns valid 256/512 PNGs.
- Download: turbo 547MB, LCM 1.6GB, custom URL — all done.
- Build: `BUILD SUCCESSFUL`.

## Not done / blockers

1. **On-device VLM** — llama.cpp/mtmd is not in the CMake build (experimental).
2. **On-device upscale** — Real-ESRGAN NCNN engine not implemented.
3. **Edits / Inpaint / Harmonize** — planned (see `docs/ONDEVICE-PLAN.md`).
4. **FFmpeg on Android** — no binary; bundled binary planned.
5. **45-min test** — the user's real recording is at
   `/sdcard/Recordings/Record/20261022_094009.wav`; run it when the device is connected.
6. **Cleanup** — ~60 scratch `fix_*.py` / `patch_*.py` files in the root.

## Build

```
cd android
.\gradlew.bat :app:assembleDebug     # BUILD SUCCESSFUL
```

## Known traps

- PowerShell 5.1 eats quotes with curl → use `--data-binary "@subor.json"`.
- `am start-foreground-service` from the shell fails → start via UI (tap 1032,498).
- Port 8101 = hex `0x1FA5`.
- Ktor 2.4.x does not exist on Maven Central → 2.3.12.
- ORT Android does not support `ConvInteger` → use the QDQ-converted UNet.
- Text encoder `input_ids` is int32 (not int64).
