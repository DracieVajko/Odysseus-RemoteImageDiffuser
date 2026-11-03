# Prism — On-device plan (Android only, offline)

Goal: **every tab works on the phone, locally and offline.** No PC, no remote
host, no network after the models are downloaded.

## Status

| Feature | Status | Engine | Notes |
|---|---|---|---|
| Transcribe / Translate | **DONE** | whisper.cpp (JNI) | 30 s chunks, auto-split |
| Generate (txt2img) | **DONE** | ONNX Runtime | LCM-Dreamshaper QDQ UNet, 256/512 px |
| Edits (img2img) | TODO | ONNX Runtime | VAE encoder already downloaded |
| Inpaint | TODO | ONNX Runtime | VAE encode + mask blend |
| Harmonize | TODO | CPU (Kotlin) | color match + seam feather |
| Upscale | TODO | ONNX Runtime or NCNN | Real-ESRGAN |
| Caption | TODO | ONNX Runtime (SmolVLM) | VLM |
| Chat (vision) | TODO | ONNX Runtime (SmolVLM) | same engine as Caption |
| FFmpeg tools | TODO | bundled ffmpeg / ffmpeg-kit | no binary today |

## 1. Edits (img2img) — reuse the LCM pipeline

Models already in `lcm-dreamshaper-int8-onnx`:
- `vae_encoder/model.onnx` (130 MB) — currently downloaded, unused.
- `unet/model.onnx` (QDQ), `text_encoder`, `vae_decoder`, `tokenizer`.

Steps:
1. Decode the input image to `FloatArray` (RGB, 0..1), resize to `size`.
2. Normalize to `[-1, 1]`, NCHW, run `vae_encoder` → `latent_sample`.
3. Multiply by `0.18215` (VAE scale).
4. `strength` (0.1..1.0): start at timestep index `round((1-strength) * steps)`.
5. Add noise at that timestep: `x_t = sqrt(a_t)*x0 + sqrt(1-a_t)*noise`.
6. Denoise the remaining steps with the prompt embedding.
7. Decode with `vae_decoder`, convert to Bitmap.
8. Unload sessions (already done).

Effort: **moderate** (~250 lines Kotlin, reuses `LcmOnnxEngine`).

## 2. Inpaint — edits + mask

Steps:
1. Same as Edits (encode + noise).
2. Build a mask tensor at latent resolution (1×1×H/8×W/8), feathered
   (Gaussian blur, radius 4–8 px).
3. After each denoise step:
   `latents = mask * latents + (1 - mask) * original_noisy_latents`.
4. Decode, composite the masked region over the original image with the
   feathered mask (avoids seams).

Effort: **moderate** (~150 lines on top of Edits).

## 3. Harmonize — CPU only (no model)

Steps:
1. Input: base image + pasted region (+ optional masks).
2. Color match: convert both to CIE LAB, compute mean/std per channel of the
   region and the surrounding background, apply `(x - mean_r) * (std_b/std_r) + mean_b`.
3. Seam fix: alpha-feather the mask (distance transform or box blur) and
   composite; optional Poisson blend if needed.
4. Optional body mask / seam mask from the request.

Effort: **low** (~150 lines, no dependencies).

## 4. Upscale — Real-ESRGAN

Options considered:
- **(a) ONNX Runtime** — export/host a Real-ESRGAN ONNX (x4plus or x4v3),
  run tiled (128 px tiles, 8 px overlap) to keep RAM bounded.
  Pros: ORT already integrated; no native build. Cons: need a verified ONNX
  model (the current catalog entry is an NCNN zip).
- **(b) NCNN Vulkan** — build Real-ESRGAN-ncnn via NDK + JNI.
  Pros: fast on Adreno. Cons: new native build, Vulkan shaders.
- **(c) CPU fallback** — Lanczos/bicubic (no AI) for 2×/4×.
  Pros: always works, tiny. Cons: not "AI upscale".

Recommendation: **(c) now + (a) next**, keep (b) as a future optimization.

Effort: (c) low (~80 lines); (a) moderate (model + tiling).

## 5. Caption / Chat — SmolVLM on ONNX

Models (`HuggingFaceTB/SmolVLM-500M-Instruct`, ONNX subfolder):
- `onnx/vision_encoder_int8.onnx` (94 MB)
- `onnx/embed_tokens_int8.onnx` (45 MB)
- `onnx/decoder_model_merged_int8.onnx` (348 MB)
- `tokenizer.json` (3.4 MB), `preprocessor_config.json`

Steps:
1. Image preprocessing: resize to 512×512, normalize per SigLIP config.
2. `vision_encoder` → image features.
3. Tokenize the prompt (`<image>` token placeholder) + text.
4. `embed_tokens` for text; merge image features at the `<image>` position.
5. Decoder loop with KV cache (merged decoder outputs `present.*`), greedy
   sampling, max 128 tokens.
6. Detokenize → caption / chat answer.
7. Unload sessions after the request.

Alternative: **llama.cpp + mtmd** (GGUF SmolVLM-500M Q8_0 + mmproj, 520 MB).
Faster at runtime, but needs a new NDK/JNI build.

Recommendation: ONNX first (no new native build), llama.cpp later if too slow.

Effort: **high** (~800–1200 lines: tokenizer, KV-cache loop, multimodal merge).

## 6. FFmpeg — bundle a binary

Options:
- **(a) ffmpeg-kit (LGPL)** — `com.arthenica:ffmpeg-kit-min` / `-audio` /
  `-https`. Adds 8–20 MB per ABI. Discontinued but still on Maven mirrors.
- **(b) Static ffmpeg per ABI** — ship `libffmpeg.so`/`ffmpeg` in `jniLibs`,
  execute via `ProcessBuilder` from `nativeLibraryDir`. ~20–30 MB.
- **(c) MediaCodec + MediaMuxer** — no extra size, but limited codecs/formats.

Recommendation: **(b)** for full format coverage and offline operation
(verify the binary/licence first), fallback (c) for audio-only operations.

Effort: low–moderate (binary + wrapper + presets from `ffmpeg_tools.py`).

## 7. Cross-cutting

- **Model catalog:** add `smolvlm-500m-onnx`, `realesrgan-x4-onnx`,
  `ffmpeg` (binary) entries with verified URLs.
- **Memory:** unload each engine after the request; `HardwareProfile` picks
  image size/threads; `largeHeap` already enabled.
- **Offline:** after download, no network calls (the model manager is the only
  network user).
- **Removed:** Local/Auto/Remote mode, remote host, `ProxyForward.kt`,
  `/webui`. All requests are served locally.

## Order of work

1. Edits (img2img) — highest value, reuses existing models.
2. Inpaint — small addition to Edits.
3. Harmonize — CPU only, quick win.
4. Upscale — CPU fallback first, then ONNX.
5. FFmpeg — bundled binary.
6. Caption/Chat — SmolVLM ONNX (largest effort).

## Status 2026-10-27

On device (verified): generation, edits, inpaint, outpaint, harmonize, upscale
(NCNN, CPU after a Vulkan self-check), caption and vision chat (llama.cpp +
libmtmd), transcription/translation (whisper.cpp), model catalog with verified
downloads, settings.

`/v1/ffmpeg/*` is implemented on device with MediaCodec/MediaMuxer (info,
images, trim, audio extraction, container remux, AAC and WAV encoding). Video
re-encode/compression is now also implemented via MediaCodec + an EGL/GL bridge
(`/v1/ffmpeg/compress`, `mp4-small` preset, ~5 MB target) in
`VideoTranscoder.kt`. What the platform still cannot do (FLAC/Opus/AVIF
encoding, VP9/WebM, H.265 transcode) answers 503 with the reason. There is no
remaining PC-only endpoint; the difference is depth: the PC can transcode to
arbitrary codecs and target sizes, the phone re-encodes to H.264 at a fixed
target.

## Transcription engines (PC): faster-whisper, Nemotron, Gemma4

Three engines are wired into the PC server; the Android app still uses
whisper.cpp only.

| Engine | How to select | Size | Notes |
|---|---|---|---|
| faster-whisper | default, model=tiny|base|small|medium|turbo|large | 75 MB - 3 GB | chunked, 5-min segments for long audio |
| NVIDIA Nemotron 3.5 ASR 0.6B | model=nemotron-asr-0.6b or PRISM_ASR_ENGINE=nemotron | 2.55 GB | FastConformer-RNNT, 40 languages (sk=28 in the prompt dictionary), native punctuation. Live: JFK 2.7 s warm vs whisper 10.3 s |
| Gemma4 E2B (post-processing) | post_process=cleanup|translate|summarize|rewrite | 10.25 GB | text-in/text-out only, not a transcription engine |

Measured on this machine (RTX 3060 6 GB, 32 GB RAM):

- Nemotron loads in ~20 s from page cache and transcribes a 11 s clip in 2.7 s.
- Gemma4 E2B does **not** fit in 6 GB VRAM; init() detects that and falls back
  to CPU. CPU load is 9-10 min, inference 13-35 s per short text.
- Gemma4 	ranslate and summarize produce correct output on CPU bf16.
  cleanup produces **mangled Slovak diacritics** (`�` -> `A?`) with bf16.
  float32 fixes precision but needs ~20 GB of RAM and starts swapping on a
  32 GB machine at ~49 % load, so it stays an opt-in
  (`PRISM_GEMMA_DTYPE=float32`).
- Correct long-term fix for Gemma4 on PC: a GGUF build with llama.cpp
  (3.1 GB Q4_K_M, minutes instead of hours). Not wired yet; it needs
  `llama-cpp-python` (a compiled dependency) and a second code path.

## Android memory limits under load (measured 2026-11-03)

A 3 GB model download running at the same time as batch image generation killed
the process: `dumpsys activity exit-info` reported
`reason=3 (LOW_MEMORY)`, `rss=1,5GB`. The downloader itself is fine (256 KB
buffer, HTTP Range resume verified: a restart continued at 2.24 GB instead of
0), so the pressure comes from the resident LCM engine plus generation alongside
the download's page cache.

Practical rule: do not run heavy generation while a large model download is in
progress. The memory watchdog (45 s idle release, 15 s poll under 1.5 GB) does
not cover this case because the engine is busy, not idle.

Also cleaned up: a 711 MB abandoned `sd15-onnx` staging directory that had
survived since 2026-10-22. Staging paths are per-model, so an abandoned download
is invisible to the catalog and never garbage-collected.

## Nemotron on Android: what is missing

The PC server can run all three Nemotron 3.5 variants. The Android app cannot
yet, and it now says so instead of substituting whisper.

Measured module structure of `nemotron-3.5-asr-streaming-0.6b` (638 M params):

| Module | Type | Params |
|---|---|---|
| `encoder` | NemotronAsrStreamingEncoder | 609.1 M |
| `encoder_projector` | Linear | 0.7 M |
| `decoder` (prediction network) | Nemotron3_5AsrRNNTDecoder | 15.4 M |
| `joint` | Nemotron3_5AsrRNNTJointNetwork | 8.4 M |
| `prompt_projector` | Nemotron3_5AsrPromptProjector | 4.5 M |

The processor emits `input_features` (B, T, 128), `attention_mask`,
`num_lookahead_tokens` and `prompt_ids`.

Plan to make it work on device (not implemented):

1. Export three ONNX graphs on PC: `encoder.onnx` (features + mask +
   prompt_ids to encoder_out), `decoder.onnx` (token to hidden),
   `joint.onnx` (encoder step + hidden to logits).
2. Dynamic-quantise the encoder to int8; the bundle lands around 700 MB to
   1 GB, comparable to the whisper model already shipped.
3. Greedy RNNT decode in Kotlin on top of `onnxruntime-android` (already a
   dependency): for each encoder step run `joint` until blank, feeding the
   emitted token back through `decoder`.
4. Add `nemotron-asr-0.6b` to `ModelCatalog.kt` pointing at the ONNX bundle and
   route `model=nemotron-asr-0.6b` in `handleAudio`.

Risks: the encoder is cache-aware streaming (attention_context_left=13,
attention_context_right=0, attention_chunk_size=12), so a single full-clip
forward may not match the reference streaming output; the export may need the
cache tensors as explicit inputs. Until then the endpoint returns 503 with this
reason.

The two 30B-A4B variants (NVFP4 21.6 GB, FP8 34.7 GB) are PC-only by design.
