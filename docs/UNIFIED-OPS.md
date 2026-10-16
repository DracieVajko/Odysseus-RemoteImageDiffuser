# UNIFIED-OPS — prevádzka unified servera (bez pushu, lokálne)

## Štart / stop (Windows, .venv)

```powershell
.\Start-Unified.ps1            # :8101 lazy, whisper fallback -> :4444
.\Start-Unified.ps1 -Remote   # + bind 0.0.0.0 (Tailscale/LAN)
.\Start-Unified.ps1 -Shim     # + legacy :4444 -> :8101 pre starých klientov
.\Stop-Unified.ps1
```

Starý `Start-Diffusion.ps1` / `Stop-Diffusion.ps1` ostáva nedotknutý (fix
branch). Nové PID súbory: `run/unified.pid`, `run/unified-shim.pid`.
Env: `UNIFIED_WHISPER_URL` (default `http://127.0.0.1:4444`),
`UNIFIED_NO_ENGINE=1` = bez diffusion registrácie (vždy placeholder).

## CLI

```powershell
.venv\Scripts\python.exe generate_image.py doctor
.venv\Scripts\python.exe generate_image.py txt2img --model DreamShaper_8 --prompt "a cabin" --output out.png
.venv\Scripts\python.exe generate_image.py upscale --input out.png --scale 2 --output out2x.png
.venv\Scripts\python.exe generate_image.py transcribe --input audio.mp3 --model small --language sk
```

## Modely

- Obrazy: `models.json` (4× SD, ~19 GB) + `download_models.py --preset odysseus-default|sd15|sdxl`.
- Upscale: `models/RealESRGAN_x4v3/RealESRGAN_x4plus.pth` (67 MB, BSD-3)
  cez `download_models.py --preset upscale`. Bez váhy = Pillow fallback
  (odpoveď hlási `"engine": "pillow"`).
- Audio: WhisperApi služba na :4444 (vlastný faster-whisper env) alebo
  `pip install faster-whisper` do `.venv` pre lokálny engine.

## Odysseus wiring

- Image base URL: `http://<host>:8101`, audio base URL: `http://<host>:8101`
  (cesty `/v1/audio/...` identické s bývalou :4444).
- Zmluva: `docs/UNIFIED-CONTRACT.md` (frozen). Starý kontrakt platí ďalej:
  `docs/ODYSSEUS-CONTRACT.md`.

## Portable balíček (Win/Mac/Linux)

V1 = repo + `.venv` + `models/` + `Start-Unified.ps1` (Win) /
`uvicorn unified_server:app` (.venv python) všade. Mac: `--cpu` cesta
automaticky (torch MPS/CPU, SD pomalé ale funkčné); Linux: rovnaký venv
postup ako Win. Plný inštalátor (PyInstaller/Briefcase) až po stabilizácii
V1 — venv má ~8 GB, modely ~19 GB, nič z toho necommitovať.

## Android

Pozri `android/README.md` + `android/contract-compat.md` a `docs/ANDROID.md`.
`./gradlew :app:assembleDebug` až na stroji s Android SDK (tu nie je).
