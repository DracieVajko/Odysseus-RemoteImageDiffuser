# Android contract-compat checklist (same-contract vs ODYSSEUS-CONTRACT)

Base: `http://<phone>:8101`. Overuje sa proti bežiacemu telefónu —
bez emulatora, ručne cez `curl`. Každý riadok: príkaz → očakávaný tvar.

> Požiadavka: `request_id` sa echoje v každej odpovedi; progress sa polluje
> počas čakania. Cold load / chýbajúce váhy → `503`, nikdy hang.

## Predpoklady

- Appka nainštalovaná, server spustený (Local alebo Auto→Local).
- `PHONE` = IP telefónu v LAN.

```bash
PHONE=192.168.1.60
```

## 1. GET /health

```bash
curl -s http://$PHONE:8101/health
```

Očakávané:

```json
{"status":"ok","model":"","current_model":"","device":"vulkan-ncnn",
 "available_models":["lcm-dreamshaper-int8","whisper-small-q5_0"],
 "loading":false,"loading_model":"","busy":false,"queue":0}
```

- [ ] `device` = `vulkan-ncnn` (nie cpu/cuda, nie nnapi)
- [ ] `busy:true` / `queue:N` = server pracuje, neposielať znova

## 2. GET /v1/models

```bash
curl -s http://$PHONE:8101/v1/models
```

- [ ] `data[].id` obsahuje `lcm-dreamshaper-int8` + `whisper-small-q5_0`
- [ ] `available:false` pred stiahnutím váh, `missing_models` ich zoznamuje
- [ ] `currently_loaded_model`, `loading`, `loading_model` prítomné

## 3. POST /v1/images/generations

```bash
curl -s http://$PHONE:8101/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a cozy mountain cabin at sunset","n":1,"size":"512x512","quality":"low","request_id":"t1"}'
```

- [ ] `200` → `{created, data:[{b64_json}], request_id:"t1"}`
- [ ] `n=5` → `400 n must be 1..4`
- [ ] `size:"1x1"` → `400` (príliš malé)
- [ ] steps cap: `quality:high` aj `steps:50` → interný clamp **4–8** (LCM režim)
- [ ] bez stiahnutého modelu → `503 will-not-load` (nie 500, nie hang)
- [ ] počas cold loadu → `503 {"error":"loading model","retry_after":30}`

## 4. POST /v1/images/edits (multipart)

```bash
curl -s http://$PHONE:8101/v1/images/edits \
  -F image=@input.png -F prompt="snowy version" -F strength=0.6 -F request_id=t2
```

- [ ] `200` → rovnaký tvar ako generations
- [ ] bez `image` partu → `400`

## 5. POST /v1/images/inpaint (JSON b64)

```bash
IMG=$(base64 -w0 input.png); MSK=$(base64 -w0 mask.png)
curl -s http://$PHONE:8101/v1/images/inpaint \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"mask\":\"$MSK\",\"prompt\":\"fix\",\"width\":512,\"height\":512,\"steps\":8,\"request_id\":\"t3\"}"
```

- [ ] `200` → `{image, elapsed, request_id}`
- [ ] bez masky → `400`
- [ ] canvas `8192+` px → `503 will-not-load` (DoS guard, nie hang)

## 6. POST /v1/images/harmonize (JSON b64)

```bash
curl -s http://$PHONE:8101/v1/images/harmonize \
  -H 'Content-Type: application/json' \
  -d "{\"image\":\"$IMG\",\"prompt\":\"evening light\",\"request_id\":\"t4\"}"
```

- [ ] `200` → `{image, elapsed, request_id}`

## 7. POST /v1/images/upscale (nové V1)

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
- [ ] prázdny súbor → `400`, nie `500`
- [ ] nepodporovaná ext (`.xyz`) → `415`
- [ ] >100 MB → `400`
- [ ] Whisper minimum pre SK = **small Q5_0** (tiny/base sa nenasadzujú)

## 9. GET /v1/images/progress/{id}

```bash
curl -s http://$PHONE:8101/v1/images/progress/t1
```

- [ ] `{id, status: running|done|error|loading|unknown, step, total, percent}`
- [ ] neznáme id → `status:unknown` (nie 500)
- [ ] `loading` = cold load prebieha, ešte žiadne steps

## 10. Prepnutie modelu za behu (single-resident)

1. `POST /v1/images/generations` → OK (image rezidentný)
2. `POST /v1/audio/transcriptions` → prvý prechod `503 loading`, po loadi OK
3. `GET /health` → `current_model` = audio model (image bol unloadnutý)

- [ ] nikdy nie sú 2 ťažké modely naraz (limit ~3–4 GB)
- [ ] concurrent generations počas loadu → `503 loading`, nie corrupt pipe

## Známé stub-obmedzenia tohto scaffoldu

- Bez stiahnutých váh vracia image/audio/upscale cesty `503 will-not-load`
  a generations vracia 1×1 stub PNG (hooky `TODO(NCNN)` / `TODO(WHISPER)`).
- ffmpeg-kit **Min (LGPL)** normalizácia (wav 16 kHz mono) je TODO po pine
  presnej LGPL-only verzie — Full/GPL sa nesmie pridať.
