# Odysseus Diffusion — Android scaffold (Kotlin + Ktor, same-contract)

Local inference server on the phone. Same contract as `docs/ODYSSEUS-CONTRACT.md`
+ `POST /v1/images/upscale` + `POST /v1/audio/*` (see `contract-compat.md`).
Base: `http://<phone>:8101`, no authentication on the LAN.

## Research verdicts (binding, from the 2026-10-15 plan)

| Decision | Reason |
|---|---|
| Image model: **LCM-Dreamshaper INT8 (NCNN/Vulkan)** = main candidate — **source NOT YET VERIFIED** | The only SD1.5 + LCM combo that actually runs on mobile within the RAM limit; the `ncnn/lcm-...` repo does not exist (404), the URL will be added only with evidence |
| Steps cap **4–8** on mobile (LCM mode) | More steps = OOM + minutes on Adreno/Mali; the server caps it |
| Backend **NCNN + Vulkan**, NOT NNAPI | NNAPI is deprecated / inconsistent on SD operators |
| Audio model: **whisper small q5_1 minimum for SK** (whisper.cpp JNI, PIN in `ModelCatalog`) | tiny/base have unusable WER for Slovak; q5_0 for small does not exist |
| ffmpeg-kit: **REMOVED** (ruling 2026-10-16) | Min artifacts pulled from Central (404), JitPack 401; decoding is handled by `AudioDecoder` (MediaCodec), zero license risk |
| SD-Turbo: non-commercial license — NOT default | Only as an opt-in experiment |
| SSD-1B / SDXL / PaliGemma-3B / SmolVLM / Moondream2: **PC-only / V2 at the earliest** | They do not fit into the ~3–4 GB mobile budget |

## RAM budget (why single-resident even on mobile)

SD INT8 (~2.5 GB) + Whisper small (~350 MB) + Real-ESRGAN (~150 MB) at once
exceed the 12 GB limit of the phone together with the other apps. `ModelManager`
therefore keeps **max 1 heavy model resident**; switching = unload + load,
concurrent requests during a load get `503 loading` (retry_after 30), not
corrupt output.

## Models are NOT in the APK

The APK contains only code. Weights are downloaded on first launch via the
`ModelDownloader` stub into `app/files/models/` (DownloadManager + checksum +
resume). Without a downloaded model the server returns `503 will-not-load`.

## Structure

```
android/
  settings.gradle.kts
  build.gradle.kts            # root (AGP versions)
  gradle.properties
  app/build.gradle.kts        # Min/LGPL deps, abiFilters
  app/src/main/AndroidManifest.xml
  app/src/main/java/eu/odysseus/diffusion/
    MainActivity.kt           # UI: Local/Remote/Auto, Start/Stop, download status
    ApiServer.kt              # Ktor routes, same paths as server :8101
    ModelManager.kt           # single-resident gate (mirror of unified/model_manager.py)
    ModelDownloader.kt        # lazy download stub (DownloadManager)
  contract-compat.md          # curl compatibility checklist
  README.md                   # this file
```

## Build (verified locally: BUILD SUCCESSFUL, 35 tasks)

```powershell
cd android
.\gradlew.bat :app:assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

The Gradle wrapper is PINned to 8.10.2 (AGP 8.5.2 cannot handle Gradle 9.x).
Important: **the Gradle JDK must be 17** (Android Studio → Settings → Build Tools
→ Gradle → Gradle JDK). With JDK 25 `:app:compileDebugKotlin` fails
(`IllegalArgumentException: 25.0.3` in the Kotlin daemon). APK ~24 MB, without models.

## Native JNI build (whisper.cpp, arm64-v8a)

```powershell
cd android
powershell -ExecutionPolicy Bypass -File fetch_third_party.ps1  # whisper.cpp v1.9.5
# + NDK r27 and CMake 3.22.1 via SDK Manager (ndkVersion PIN in app/build.gradle.kts)
.\gradlew.bat :app:assembleDebug   # also builds libwhisper.so into the APK
```

`app/src/main/cpp/`: `whisper_jni.cpp` (thin bridge over the stable C API —
init_from_file/full/segments/free) + `CMakeLists.txt` (whisper.cpp as a
static lib, output `libwhisper.so`). CMake is wired in **only when the NDK is
present** (`SDK/ndk/<ver>` or `ANDROID_NDK_HOME`), otherwise a pure Kotlin build
with a warning — `WhisperLib` then takes the stub path. For a full audio build
install **NDK r27 + CMake 3.22.1** (SDK Manager) and rebuild.

## Modes (Local / Remote / Auto)

- **Local** — inference on the phone (NCNN/Vulkan + whisper.cpp JNI).
- **Remote** — the phone is a thin client to the PC server `http://<pc>:8101`
  (same contract, no change to the calls).
- **Auto** (default) — try Local, on `503 will-not-load` / OOM fall back to
  the Remote host from settings.

## Ktor routes (full match with the server)

GET /health, GET /v1/models, POST /v1/images/generations|edits,
POST /v1/images/inpaint|harmonize|upscale, POST /v1/audio/transcriptions|translations,
GET /v1/images/progress/{id}
