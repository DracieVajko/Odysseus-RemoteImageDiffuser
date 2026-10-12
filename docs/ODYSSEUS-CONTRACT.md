# Odysseus integration contract (server side, frozen)

Base: `http://<host>:8101`. No auth on LAN. All image routes accept `model`
(registry id, empty = default) and `request_id` (echoed back in every response;
poll it while waiting).

## Endpoints

| Method | Path | In | Out |
|---|---|---|---|
| GET | `/health` | — | `{status, model, current_model, device, available_models, loading, loading_model, busy, queue}` (`device`: cuda/cpu/None) |
| GET | `/v1/models` | — | `{data:[{id, object, owned_by, available, guessed}], currently_loaded_model, loading, loading_model, missing_models}` |
| GET | `/v1/images/progress/{request_id}` | — | `{id, status: running\|done\|error\|loading\|unknown, step, total, percent}` (`loading` = cold load in progress, no steps yet) |
| POST | `/v1/images/generations` | JSON `{model, prompt, n 1..4, size WxH, quality, request_id}` | `{created, data:[{b64_json}], request_id}` |
| POST | `/v1/images/edits` | multipart `image + prompt, model, n, size, quality, strength 0.1..1, request_id` | same as generations |
| POST | `/v1/images/inpaint` | JSON `{image b64, mask b64, prompt, width, height, steps, strength, feather, model, request_id}` | `{image b64, elapsed, request_id}` |
| POST | `/v1/images/harmonize` | JSON `{image b64, prompt, color_match, seam_fix, body_mask?, seam_mask?, mask?, strength?, max_side, steps, model, request_id}` | `{image b64, elapsed, request_id}` |
| GET | `/webui` | — | single-file HTML UI (humans only, same-origin) |

## Quality → steps

`low=4`, `medium`=profile default (25), `high=30`, `auto=12`. Recommended
default for clients: `medium`.

## Status codes

- `200` + image: success (check the PNG, not just the code)
- `400`: bad input (`Size 1x1 too small`, `n must be 1..4`, bad base64/image)
- `503 {"error":"loading model","loading_model":..., "retry_after":30}`:
  cold load in progress — wait and retry, do NOT treat as failure
- `503 {"error":"will not load",...}`:
  model fits neither VRAM nor RAM right now — do NOT retry blindly; free VRAM
  (close other GPU apps) or RAM first. Answered immediately, no hanging.
  Same shape for OOM during load *or* inference (e.g. absurd canvas).
- `500`: real server failure, see server logs (never for queueing/validation)

## Timing guidance for clients

- Cold load (model switch): SD1.5 ~1–3 min, SDXL ~3–6 min. Only ONE model is
  resident — every switch costs a full load.
- VRAM short? The server loads on CPU from RAM automatically when it fits
  there (works, MUCH slower — close LM Studio for full speed). Fits nowhere:
  immediate `will not load`, no hanging.
- Canvas caps: generations keep a 2048px VRAM-fit cap; edits/inpaint/harmonize
  accept any sane size (DoS guard 8192px) and force CPU offload on big
  canvases instead of refusing. OOM anywhere → 503, never a hang.
- Recommended client timeouts: generations/edits/inpaint ≥300 s (1024px/high
  up to ~420 s on 6 GB VRAM). Progress-poll `/v1/images/progress/{id}` for UX.
- `busy:true` / `queue:N` in `/health` = server works, don't resend.
