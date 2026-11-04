# Administrator setup (one-time)

Everything here needs an **elevated** PowerShell (Run as Administrator). The
non-elevated paths are already done: the server runs on `http://127.0.0.1:8101`
from the user's Startup folder.

---

## 1. Turn Prism into a real Windows service and retire the legacy pair

```powershell
cd D:\AI\DiffusionServer
powershell -ExecutionPolicy Bypass -File .\pc\scripts\install-prism-service.ps1 -RemoveLegacy
```

This does, in order:

| Step | Effect |
|---|---|
| `nssm install Prism C:\Python314\python.exe` | new service entry |
| `AppParameters = -m uvicorn unified_server:app --host 0.0.0.0 --port 8101` | one endpoint for everything |
| `AppDirectory = D:\AI\DiffusionServer\pc` | imports `unified_server` |
| `Start = SERVICE_AUTO_START` | starts at boot, before logon |
| `AppStdout/AppStderr = pc\logs\service-*.log`, 10 MB rotation | logs you can read |
| `Stop-Service` + `Set-Service -StartupType Disabled` on `OdysseusDiffusion` and `whisper-api` | no port conflicts, no second whisper |

The legacy `whisper-api` had `StartMode = Auto` and would keep binding port 4444
on every boot; Prism has faster-whisper built in, so it is redundant.

### Remove the non-elevated autostart afterwards

Once the service exists, delete the Startup shortcut so you do not run two
servers:

```powershell
Remove-Item "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\Prism.vbs" -ErrorAction SilentlyContinue
```

## 2. Verify

```powershell
Get-Service Prism, OdysseusDiffusion, whisper-api | Format-Table Name, Status, StartType
Invoke-RestMethod http://127.0.0.1:8101/health | Select-Object status, current_model, audio_loaded
```

Expected: `Prism = Running/Automatic`, the two legacy ones `Stopped/Disabled`,
`current_model = $null`, `audio_loaded = False` (nothing loaded until a request
arrives).

## 3. Point Odysseus at the single base URL

```
http://<this-machine>:8101/v1
```

Everything the old two services served is here:

| Old | New |
|---|---|
| `OdysseusDiffusion` :8101 images | `POST /v1/images/generations`, `/edits`, `/inpaint`, `/outpaint`, `/upscale` |
| `whisper-api` :4444 | `POST /v1/audio/transcriptions` (`model=small|turbo|large|nemotron-asr-0.6b`), `/translations` |

## 4. Later: drop the FP8 variant once NVFP4 is proven

```powershell
# stop it being offered first (edit the catalog), then delete the weights
Remove-Item "D:\AI\DiffusionServer\pc\models\nemotron-transcribe-30b-a4b-fp8" -Recurse -Force
```

The catalog entry lives in `pc\download_models.py` under `AUDIO_TASKS`; delete
the `nemotron-transcribe-30b-a4b-fp8` block so the Models tab stops offering it.

## 5. Service control cheatsheet

```powershell
Restart-Service Prism
Stop-Service Prism
Get-Content D:\AI\DiffusionServer\pc\logs\service-err.log -Tail 50
```

To uninstall everything:

```powershell
powershell -ExecutionPolicy Bypass -File .\pc\scripts\install-prism-service.ps1 -Uninstall
```
