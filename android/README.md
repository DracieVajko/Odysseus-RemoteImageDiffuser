# Odysseus Diffusion — Android scaffold (Kotlin + Ktor, same-contract)

Lokálny inference server na telefóne. Rovnaká zmluva ako `docs/ODYSSEUS-CONTRACT.md`
+ `POST /v1/images/upscale` + `POST /v1/audio/*` (viď `contract-compat.md`).
Base: `http://<phone>:8101`, žiadna autentifikácia na LAN.

## Research verdikty (záväzné, z plánu 2026-10-15)

| Rozhodnutie | Dôvod |
|---|---|
| Image model: **LCM-Dreamshaper INT8 (NCNN/Vulkan)** = hlavný kandidát | Jediný SD1.5 + LCM combo, ktorý reálne beží na mobile v RAM limite |
| Steps cap **4–8** na mobile (LCM režim) | Viac steps = OOM + minúty na Adreno/Mali; server capuje |
| Backend **NCNN + Vulkan**, NIE NNAPI | NNAPI je deprecated / nekonzistentné na SD operátoroch |
| Audio model: **whisper small Q5_0 minimum pre SK** (whisper.cpp JNI) | tiny/base majú nepoužiteľné WER pre slovenčinu |
| ffmpeg-kit **Min (LGPL) only**, NIE Full/GPL | Full/GPL sťahuje x264/x265 → licenčný problém v APK |
| SD-Turbo: nekomerčná licencia — NIE default | Len ako opt-in experiment |
| SSD-1B / SDXL / PaliGemma-3B / SmolVLM / Moondream2: **PC-only / až V2** | Nezmestia sa do ~3–4 GB mobilného budgetu |

## RAM budget (prečo single-resident aj na mobile)

SD INT8 (~2.5 GB) + Whisper small (~350 MB) + Real-ESRGAN (~150 MB) naraz
prekročia limit 12 GB telefónu s ostatnými appkami. `ModelManager` preto drží
**max 1 ťažký model rezidentný**; prepnutie = unload + load, concurrent
požiadavky počas loadu dostanú `503 loading` (retry_after 30), nie corrupt výstup.

## Modely NIE sú v APK

APK obsahuje len kód. Váhy sa sťahujú pri prvom spustení cez
`ModelDownloader` stub do `app/files/models/` (DownloadManager + checksum +
resume). Bez stiahnutého modelu vracia server `503 will-not-load`.

## Štruktúra

```
android/
  settings.gradle.kts
  build.gradle.kts            # root (AGP versions)
  gradle.properties
  app/build.gradle.kts        # Min/LGPL deps, abiFilters
  app/src/main/AndroidManifest.xml
  app/src/main/java/eu/odysseus/diffusion/
    MainActivity.kt           # UI: Local/Remote/Auto, Start/Stop, download stav
    ApiServer.kt              # Ktor routes, rovnaké cesty ako server :8101
    ModelManager.kt           # single-resident gate (zrkadlo unified/model_manager.py)
    ModelDownloader.kt        # lazy download stub (DownloadManager)
  contract-compat.md          # curl checklist kompatibility
  README.md                   # tento súbor
```

## Build (až na stroji s Android SDK)

```bash
cd android
./gradlew :app:assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

Bez Android SDK / emulatora sa build neoveruje — kompatibilitu overuje
`contract-compat.md` (curl tvary proti bežiacemu telefónu).

## Režimy (Local / Remote / Auto)

- **Local** — inferencia na telefóne (NCNN/Vulkan + whisper.cpp JNI).
- **Remote** — telefón je tenký klient na PC server `http://<pc>:8101`
  (rovnaká zmluva, žiadna zmena volaní).
- **Auto** (default) — skús Local, pri `503 will-not-load` / OOM fallback na
  Remote host z nastavení.

## Ktor cesty (plná zhoda so serverom)

GET /health, GET /v1/models, POST /v1/images/generations|edits,
POST /v1/images/inpaint|harmonize|upscale, POST /v1/audio/transcriptions|translations,
GET /v1/images/progress/{id}
