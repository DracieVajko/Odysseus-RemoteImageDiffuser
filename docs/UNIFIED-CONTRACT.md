# Unified contract (PC :8101, Android :8101) — frozen pre Odysseus

Base: `http://<host>:8101`. LAN-only, bez auth. Všetky image routy berú `model`
(registry id, prázdne = default) a `request_id` (echo späť).

Legacy: staré `WhisperApi :4444/v1/audio/*` ostáva ako shim na `:8101/v1/audio/*`
(tvar request/response identický). Nový kód volá vždy `:8101`.

## Endpointy

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
| POST | `/v1/audio/transcriptions` | multipart `file|audio + model, device, language` | `{text, language}` alebo `text/plain` |
| POST | `/v1/audio/translations` | same | `{text}` |
| GET | `/webui` | — | unified HTML (Generate/Edit-Upscale/Transcribe) |

## Status kódy (zhodné s ODYSSEUS-CONTRACT + audio)

- `200` + dáta: úspech.
- `400`: zlý vstup (missing image, Empty file, zlý scale).
- `415`: nepodporovaný audio formát. `413`: >100 MB.
- `503 {"error":"loading model","loading_model":..., "retry_after":30}`: cold load — čakať, nie failure.
- `503 {"error":"will not load",...}`: nevojde sa do VRAM ani RAM, alebo canvas >8192px — neuvoľní sa hangom.
- `500`: reálna chyba servera.

## Limity V1

- Vždy max 1 ťažký model (SD *alebo* Whisper). Prepnutie = unload + load.
- Upscale default `RealESRGAN_x4v3`, tile processing, cap 8192px.
- Audio: 16 kHz mono normalizácia cez ffmpeg (ak je), inak passthrough wav.
- Mobile: steps clamp 4–8 (thermal throttle po 3. generácii −40%), Vulkan (nie NNAPI).
