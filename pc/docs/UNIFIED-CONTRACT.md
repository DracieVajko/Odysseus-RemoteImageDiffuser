# Unified contract (PC :8101, Android :8101) — frozen for Odysseus

Base: `http://<host>:8101`. LAN-only, no auth. All image routes take `model`
(registry id, empty = default) and `request_id` (echoed back).

Legacy: the old `WhisperApi :4444/v1/audio/*` remains as a shim to `:8101/v1/audio/*`
(request/response shape identical). New code always calls `:8101`.

## Endpoints

| Method | Path | In | Out |
|---|---|---|---|
| GET | `/health` | — | `{status, model, current_model, device, available_models, loading, loading_model, busy, queue, audio_model, audio_loaded}` |
| GET | `/v1/models` | — | `{data:[{id...}], currently_loaded_model, loading, loading_model, missing_models, default_image_model, default_audio_model, audio_models}` |
| GET | `/v1/images/progress/{request_id}` | — | `{id, status: running\|done\|error\|loading\|unknown, step, total, percent}` |
| POST | `/v1/images/generations` | JSON `{model, prompt, n 1..4, size WxH, quality, request_id}` | `{created, data:[{b64_json}], request_id}` |
| POST | `/v1/images/edits` | multipart `image + prompt, model, n, size, quality, strength, request_id` | same |
| POST | `/v1/images/inpaint` | JSON `{image b64, mask b64, prompt, strength, model, request_id}` | `{image b64, elapsed, request_id}` |
| POST | `/v1/images/harmonize` | JSON `{image b64, prompt, strength, model, request_id}` | `{image b64, elapsed, request_id}` |
| POST | `/v1/images/upscale` | JSON `{image b64, scale 2/3/4, model, request_id}` | `{image b64, elapsed, model, scale, request_id}` |
| POST | `/v1/images/caption` | JSON `{image b64, prompt, max_tokens, request_id}` | `{text, elapsed, engine, request_id}` (V2, SmolVLM-500M) |
| POST | `/v1/chat/completions` | OpenAI vision subset `{model, messages:[{role, content:str\|[{type:text\|image_url}]}]}` | `{choices:[{message:{content}}]}` (V2, 503 without weights) |
| GET | `/v1/models/catalog` | — | `{models:[{id, kind, label, size_gb, installed, installed_gb, install_state, install_progress, install_error}]}` |
| POST | `/v1/models/install` | JSON `{id}` | `{id, state}` (download in the background; progress in the catalog) |
| POST | `/v1/models/uninstall` | JSON `{id}` | `{id, removed}` (path guard: only `models/`) |
| GET | `/v1/settings` | — | `{version, port, whisper_url, models_dir, default_image_model, audio_models, engines:{torch,cuda,ffmpeg,ffprobe,ytdlp}}` |
| GET | `/v1/ffmpeg/presets` | — | `{video:{...}, audio:{...}, resize:{...}}` |
| POST | `/v1/ffmpeg/{convert-video\|convert-audio\|extract-audio\|trim\|compress\|info}` | multipart | `{file\|info, filename, request_id, info:{duration,size,format}}` |
| POST | `/v1/audio/transcriptions` | multipart `file|audio + model, device, language` | `{text, language}` or `text/plain` |
| POST | `/v1/audio/translations` | same | `{text}` |
| GET | `/webui` | — | unified HTML (Generate/Edit-Upscale/Transcribe) |

## Status codes (identical to ODYSSEUS-CONTRACT + audio)

- `200` + data: success.
- `400`: bad input (missing image, Empty file, bad scale).
- `415`: unsupported audio format. `413`: >100 MB.
- `503 {"error":"loading model","loading_model":..., "retry_after":30}`: cold load — wait, not a failure.
- `503 {"error":"will not load",...}`: does not fit into VRAM or RAM, or canvas >8192px — it fails fast instead of hanging.
- `500`: real server error.

## V1 limits

- Always at most 1 heavy model (SD *or* Whisper). Switching = unload + load.
- Upscale default `RealESRGAN_x4v3`, tile processing, cap 8192px.
- Audio: 16 kHz mono normalization via ffmpeg (if available), otherwise wav passthrough.
- Mobile: steps clamp 4–8 (thermal throttle after the 3rd generation −40%), Vulkan (not NNAPI).
