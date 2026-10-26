# Project instructions (Prism)

Language: **English only.** All responses, code comments, documentation, commit
messages, UI strings, log messages and error messages must be in English. No
Slovak (or any other language) in the codebase or user-facing output.

Exception: Android UI translations live in `res/values-sk`, `values-cs`,
`values-de` (the base `values/strings.xml` stays English).

## Architecture (locked)

Prism is **two independent products**:

1. **Android app** — native UI only, **no WebUI**, **no PC/remote mode**.
   Every feature runs **on the phone, locally and offline**: generate, edits,
   inpaint, harmonize, upscale, caption, chat, transcribe, translate, ffmpeg.
   There is no Local/Auto/Remote switch and no remote host field.
2. **PC server** — `pc/unified_server.py` + `pc/webui_unified.html`, fully
   separate.

Both expose the same OpenAI-compatible API so external AI clients can attach
over LAN, but the Android app never depends on the PC.

### Repository layout

| Path | Contents |
|---|---|
| `pc/` | PC product: server, `unified/`, WebUI, models, tests, scripts, docs |
| `android/` | Android product: Gradle app, native engines, tests, tools |
| `docs/` | shared project docs (architecture, checklist, checkpoints, plan) |
| `testdata/` | shared fixtures used by the PC/Android test drivers |

A user only needs `pc/` for the PC server or `android/` for the app. Shared
folders are development material, not runtime dependencies.

If a feature cannot run on Android yet, mark it **experimental**, keep the
endpoint honest (503 with a clear reason), and document the plan in
`docs/ONDEVICE-PLAN.md`. Never fake success.

## Endpoints (contract)

`GET /health`, `GET /v1/models`, `GET|POST /v1/models/*`,
`POST /v1/images/generations|edits|inpaint|harmonize|upscale|caption`,
`POST /v1/chat/completions`, `POST /v1/audio/transcriptions|translations`,
`GET /v1/ffmpeg/presets`, `POST /v1/ffmpeg/{op}`, `GET /v1/settings`.

## Conventions

- Models are downloaded lazily via `ModelDownloader` (resume, staging, `.READY`
  manifest). Never bundle weights in the APK.
- Catalog entries must have verified (HTTP 200) URLs.
- Android keeps the LCM engine resident and releases it after 10 minutes
  without image work (one ORT run at a time); the PC unloads on idle/switch.
- All UI text in English. All buttons must be reachable and tested.
- The LCM UNet must be **QDQ** format (ORT Android has no `ConvInteger`).
  Conversion script: `pc/tools/convert_lcm_qdq.py`.

## Instructions

- **Checklist:** `docs/CHECKLIST.md` — every task and its status.
- **Checkpoints:** `docs/CHECKPOINTS.md` — decisions and verified states.
- **Architecture:** `docs/ARCHITECTURE.md` — end-to-end PC and Android flow.
- **On-device plan:** `docs/ONDEVICE-PLAN.md` — how each feature will work
  locally, effort, and order of work.
- **Android details:** `android/docs/ANDROID.md`.
- Update the checklist after every meaningful change.

## Build

```
cd android
.\gradlew.bat :app:assembleDebug
```

## Run (PC)

```
cd pc
python -m uvicorn unified_server:app --host 0.0.0.0 --port 8101
```
