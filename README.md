# Prism

Two independent products live in this repository. You only need the folder for
the one you want to run.

## Start here (PC — server + WebUI)

1. Double-click **`Start-Prism.bat`** in this folder.
2. The browser opens the WebUI automatically.
   - `http://127.0.0.1:8101/webui` normally.
   - If port 8101 is still used by the old `OdysseusDiffusion` service, the
     launcher starts Prism on **8102** and opens that URL instead.
3. Stop it with **`Stop-Prism.bat`**.

The WebUI has Generate, Edit/Inpaint, Upscale, Transcribe, Caption/Chat and
FFmpeg tools. If a feature is not available on this machine it says so
(503 will-not-load) instead of faking output.

Android is separate: build/install `android/` (see below).

| Product | Folder | What it is |
|---|---|---|
| **PC server + WebUI** | `pc/` | FastAPI server with an OpenAI-compatible API, ONNX/Diffusers image generation, LCM inpaint, Harmonize, Real-ESRGAN upscale, SmolVLM caption/chat, FFmpeg presets. GPU via CUDA or DirectML. |
| **Android app** | `android/` | Native Prism app with an on-device Ktor server (ONNX LCM, whisper.cpp, inpaint, harmonize). No WebUI, no PC dependency. |

## Pick one

**PC only** — copy/download `pc/`, then:

```powershell
cd pc
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_models.py --preset odysseus-default
python -m uvicorn unified_server:app --host 0.0.0.0 --port 8101
```

Open `http://127.0.0.1:8101/webui`. Full guide: `pc/docs/UNIFIED-OPS.md`.

**Android only** — install the APK, or build it yourself.

The APK is **not** in git: `android/app/build/` is gitignored, because a 71 MB
binary does not belong in version control and any commit would go stale. There
are three ways to get one:

1. **GitHub Releases (easiest for users).** Push a tag and the
   `Release` workflow builds and attaches the APK:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```
   Then download `prism-v0.1.0-release.apk` from
   `https://github.com/<owner>/<repo>/releases`.
2. **CI artifact.** Every push to `master` produces `prism-debug-apk` under
   the run's *Artifacts* section (`.github/workflows/ci.yml`).
3. **Build locally.**
   ```powershell
   cd android
   .\gradlew.bat :app:assembleDebug     # debug, debug-key signed
   .\gradlew.bat :app:assembleRelease   # minified; debug-key signed unless a keystore is configured
   adb install -r app\build\outputs\apk\debug\app-debug.apk
   ```

To sign a release properly, create `android/keystore.properties`:
```properties
storeFile=/absolute/path/prism-release.jks
storePassword=...
keyAlias=prism
keyPassword=...
```
(or set `PRISM_KEYSTORE_FILE`, `PRISM_KEYSTORE_PASSWORD`, `PRISM_KEY_ALIAS`,
`PRISM_KEY_PASSWORD`). Without it the release APK is signed with the debug key,
which is installable by side-loading but not publishable on Play.

On the phone: allow "Install unknown apps" for your browser/file manager, open
the APK, install. Android 10+ (arm64-v8a). Models are downloaded on the device
from the in-app model catalog; the app is fully offline afterwards. Guide:
`android/README.md`.

## Shared development material (not needed at runtime)

- `docs/` — architecture, checklist, checkpoints, on-device plan.
- `testdata/` — reference image/mask/audio/video used by the test suites.
- `pc/tools/`, `android/tests/` — end-to-end test drivers.
- `AGENTS.md` — working rules for agents/contributors.

## Contracts

Both products expose the same core paths: `GET /health`, `GET /v1/models`,
`POST /v1/images/generations|edits|inpaint|harmonize|upscale|caption`,
`POST /v1/chat/completions`, `POST /v1/audio/transcriptions|translations`,
`GET /v1/ffmpeg/presets`, `POST /v1/ffmpeg/{op}`, `GET /v1/settings`.
Unimplemented features answer `503 will-not-load`, never fake success.

## Security

LAN-only, no authentication. Do not expose the ports to the internet; bind
Tailscale or your LAN. The NSFW safety checker is disabled in the launchers.

## License

MIT — see `LICENSE`.

---

## Support this work

If you find this project useful, consider supporting its ongoing development, maintenance, and improvements.

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub-ea4aaa?logo=github)](https://github.com/sponsors/DracieVajko)
