# Odysseus Diffusion — Android scaffold (Kotlin + Ktor, same-contract)

Lokálny inference server na telefóne. Rovnaká zmluva ako `docs/ODYSSEUS-CONTRACT.md`
+ `POST /v1/images/upscale` + `POST /v1/audio/*` (viď `contract-compat.md`).
Base: `http://<phone>:8101`, žiadna autentifikácia na LAN.

## Research verdikty (záväzné, z plánu 2026-10-15)

| Rozhodnutie | Dôvod |
|---|---|
| Image model: **LCM-Dreamshaper INT8 (NCNN/Vulkan)** = hlavný kandidát — **zdroj ZATIAĽ NEOVERENÝ** | Jediný SD1.5 + LCM combo, ktorý reálne beží na mobile v RAM limite; `ncnn/lcm-...` repo neexistuje (404), URL sa doplní až s dôkazom |
| Steps cap **4–8** na mobile (LCM režim) | Viac steps = OOM + minúty na Adreno/Mali; server capuje |
| Backend **NCNN + Vulkan**, NIE NNAPI | NNAPI je deprecated / nekonzistentné na SD operátoroch |
| Audio model: **whisper small q5_1 minimum pre SK** (whisper.cpp JNI, PIN v `ModelCatalog`) | tiny/base majú nepoužiteľné WER pre slovenčinu; q5_0 pre small neexistuje |
| ffmpeg-kit: **VYRADENÝ** (ruling 2026-10-16) | Min artefakty stiahnuté z Central (404), JitPack 401; dekódovanie rieši `AudioDecoder` (MediaCodec), nulové licenčné riziko |
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

## Build (overené lokálne: BUILD SUCCESSFUL, 35 taskov)

```powershell
cd android
.\gradlew.bat :app:assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

Gradle wrapper je PINnutý na 8.10.2 (AGP 8.5.2 nezvláda Gradle 9.x).
Dôležité: **Gradle JDK musí byť 17** (Android Studio → Settings → Build Tools
→ Gradle → Gradle JDK). S JDK 25 padá `:app:compileDebugKotlin`
(`IllegalArgumentException: 25.0.3` v Kotlin daemone). APK ~24 MB, bez modelov.

## Natívny JNI build (whisper.cpp, arm64-v8a)

```powershell
cd android
powershell -ExecutionPolicy Bypass -File fetch_third_party.ps1  # whisper.cpp v1.9.5
# + NDK r27 a CMake 3.22.1 cez SDK Manager (ndkVersion PIN v app/build.gradle.kts)
.\gradlew.bat :app:assembleDebug   # postaví aj libwhisper.so do APK
```

`app/src/main/cpp/`: `whisper_jni.cpp` (tenký most nad stabilným C API —
init_from_file/full/segments/free) + `CMakeLists.txt` (whisper.cpp ako
statická lib, výstup `libwhisper.so`). CMake sa zapojí **len keď je NDK**
(`SDK/ndk/<ver>` alebo `ANDROID_NDK_HOME`), inak čistý Kotlin build
s varovaním — `WhisperLib` vtedy ide stub cestou. Pre plný audio build
nainštaluj **NDK r27 + CMake 3.22.1** (SDK Manager) a prebuilduj.

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
