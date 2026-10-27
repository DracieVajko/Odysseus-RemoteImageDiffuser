# Android contract-compat checklist (same-contract vs ODYSSEUS-CONTRACT)

Base: `http://<phone>:8101`. Verified against a running phone —
no emulator, manually via `curl`. Each line: command → expected shape.

> Requirement: `request_id` is echoed in every response; progress is polled
> while waiting. Cold load / missing weights → `503`, never a hang.

## Prerequisites

- App installed, server running (Local or Auto→Local).
- `PHONE` = the phone's IP in the LAN.

```bash
# The phone's LAN address (router DHCP lease, or `adb shell ip addr`).
PHONE=<phone-ip>
```

## 1. GET /health

```bash
curl -s http://$PHONE:8101/health
```

Expected:

```json
{"status":"ok","model":"","current_model":"","device":"vulkan-ncnn",
 "available_models":["lcm-dreamshaper-int8","whisper-small-q5_0"],
 "loading":false,"loading_model":"","busy":false,"queue":0}
```

- [ ] `device` = `vulkan-ncnn` (not cpu/cuda, not nnapi)
- [ ] `busy:true` / `queue:N` = the server is working, do not send again

## 2. GET /v1/models

```bash
curl -s http://$PHONE:8101/v1/models
```

- [ ] `data[].id` contains `lcm-dreamshaper-int8` + `whisper-small-q5_0`
- [ ] `available:false` before the weights are downloaded, `missing_models` lists them
- [ ] `currently_loaded_model`, `loading`, `loading_model` present

## 3. POST /v1/images/generations

```bash
curl -s http://$PHONE:8101/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a cozy mountain cabin at sunset","n":1,"size":"512x512","quality":"low","request_id":"t1"}'
```

- [ ] `200` → `{created, data:[{b64_json}], request_id:"t1"}`
- [ ] `n=5` → `400 n must be 1..4`
- [ ] `size:"1x1"` → `400` (too small)
- [ ] steps cap: `quality:high` and `steps:50` → internal clamp **4–8** (LCM mode)
- [ ] without a downloaded model → `503 will-not-load` (not 500, not a hang)
- [ ] during cold load → `503 {"error":"loading model","retry_after":30}`

## 4. POST /v1/images/edits (multipart)

```bash
curl -s http://$PHONE:8101/v1/images/edits \
  -F image=@input.png -F prompt="snowy version" -F strength=0.6 -F request_id=t2
```

- [ ] `200` → same shape as generations
- [ ] without the `image` part → `400`

## 5. POST /v1/images/inpaint (JSON b64)

```bash
IMG=$(base64 -w0 input.png); MSK=$(base64 -w0 mask.png)
curl -s http://$PHONE:8101/v1/images/inpaint \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"mask\":\"$MSK\",\"prompt\":\"fix\",\"width\":512,\"height\":512,\"steps\":8,\"request_id\":\"t3\"}"
```

- [ ] `200` → `{image, elapsed, request_id}`
- [ ] without a mask → `400`
- [ ] canvas `8192+` px → `503 will-not-load` (DoS guard, not a hang)

## 6. POST /v1/images/harmonize (JSON b64)

```bash
curl -s http://$PHONE:8101/v1/images/harmonize \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"prompt\":\"evening light\",\"request_id\":\"t4\"}"
```

- [ ] `200` → `{image, elapsed, request_id}`

## 7. POST /v1/images/upscale (new in V1)

```bash
curl -s http://$PHONE:8101/v1/images/upscale \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"scale\":2}"
```

- [ ] `200` → `{image, elapsed}` (default Real-ESRGAN-x4v3 NCNN)
- [ ] `scale:8` → `400`

## 8. POST /v1/audio/transcriptions + /translations (multipart)

```bash
curl -s http://$PHONE:8101/v1/audio/transcriptions -F audio=@rec.m4a -F language=sk
curl -s http://$PHONE:8101/v1/audio/translations   -F audio=@rec.m4a
```

- [ ] `200` → `{text, language, ...}`
- [ ] empty file → `400`, not `500`
- [ ] unsupported ext (`.xyz`) → `415`
- [ ] >100 MB → `400`
- [ ] Whisper minimum for SK = **small Q5_0** (tiny/base are not deployed)

## 9. GET /v1/images/progress/{id}

```bash
curl -s http://$PHONE:8101/v1/images/progress/t1
```

- [ ] `{id, status: running|done|error|loading|unknown, step, total, percent}`
- [ ] unknown id → `status:unknown` (not 500)
- [ ] `loading` = cold load in progress, no steps yet

## 10. Model switch at runtime (single-resident)

1. `POST /v1/images/generations` → OK (image resident)
2. `POST /v1/audio/transcriptions` → first pass `503 loading`, OK after the load
3. `GET /health` → `current_model` = audio model (the image model was unloaded)

- [ ] there are never 2 heavy models at once (limit ~3–4 GB)
- [ ] concurrent generations during a load → `503 loading`, not a corrupt pipe

## Known stub limitations of this scaffold

- Without downloaded weights the image/audio/upscale routes return `503 will-not-load`
  and generations returns a 1×1 stub PNG (hooks `TODO(NCNN)` / `TODO(WHISPER)`).
- ffmpeg-kit **Min (LGPL)** normalization (wav 16 kHz mono) is TODO after pinning
  the exact LGPL-only version — Full/GPL must not be added.
