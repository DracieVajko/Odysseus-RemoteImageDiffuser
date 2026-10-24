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
