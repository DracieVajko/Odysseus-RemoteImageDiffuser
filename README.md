# Prism

Two independent products live in this repository. You only need the folder for
the one you want to run.

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

**Android only** — copy/download `android/`, then:

```powershell
cd android
.\gradlew.bat :app:assembleDebug
adb install -r app\build\outputs\apk\debug\app-debug.apk
```

Models are downloaded on the device from the in-app model catalog. Guide:
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
