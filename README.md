# Odysseus RemoteImageDiffuser

A lazy-loading, OpenAI-compatible Stable Diffusion image server for Windows
and Linux with NVIDIA GPUs. Built as an **optional remote image backend for
[Odysseus](https://github.com/)** — generate outside the Docker host over
Tailscale or plain LAN. Also works standalone (WebUI, CLI).

- Lazy loading: nothing loaded at startup, VRAM stays free; the first request
  loads the model, it stays resident (switching = unload + load)
- Models live in `models/` (~19 GB, downloaded by script — **never committed**)
- `GET /health`, `GET /v1/models`, `POST /v1/images/generations|edits|inpaint|harmonize`
- Single-file WebUI at `/webui`, CLI (`generate_image.py`), `doctor` diagnostics

## Requirements

- Windows 10/11 or Linux, NVIDIA GPU (tested on 6 GB; 8 GB+ better for SDXL)
- Python 3.12, ~25 GB free disk (venv ~8 GB + models ~19 GB)
- Tailscale (optional — plain LAN IP works the same way)

## Quickstart (5 minutes)

```powershell
cd <path>\Odysseus-RemoteImageDiffuser
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.12.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
.venv\Scripts\python.exe download_models.py --preset odysseus-default
.venv\Scripts\python.exe generate_image.py doctor
```

Start (lazy, port 8101, reachable remotely):

```powershell
.\Start-Diffusion.ps1 -Lazy -Remote
```

Verify: http://127.0.0.1:8101/webui — type a prompt, generate.

Full install as a service + firewall + Odysseus wiring: see **INSTALL.md**.
Operations, logs, restarts: **OPERATIONS.md**. When something breaks:
**TROUBLESHOOTING.md**. API contract for integrations:
**docs/ODYSSEUS-CONTRACT.md**.

## Models

| ID | Type | Size | Source |
|---|---|---|---|
| DreamShaper_8 | SD1.5 general | 2.7 GB | HF `Lykon/dreamshaper-8` (fp16 repo) |
| AbsoluteReality_1.8 | SD1.5 photoreal | 2.0 GB | HF `Lykon/AbsoluteReality` (single file) |
| AnimagineXL_3.1 | SDXL anime | 6.5 GB | HF `cagliostrolab/...` (single file) |
| JuggernautXL_v9 | SDXL photoreal | 6.6 GB | HF `RunDiffusion/...` (single file) |

Drop your own `.safetensors` into `models/<Name>/` — the server auto-detects it
after restart with a guessed profile (`guessed:true`); confirm the profile in
`models.json` for permanent use (format in OPERATIONS.md). Model licenses: see
the repo cards on HuggingFace (weights are downloaded, not part of this repo).

## Performance (6 GB card, approximate)

Cold load: SD1.5 ~1–3 min, SDXL ~3–6 min. Warm: 512px/low in seconds,
1024px/high in minutes. Recommended: 768px + medium. If VRAM overflows you have
two options: (a) smaller image/fewer steps, VRAM-only (faster, weaker), or
(b) CPU offload spilling into RAM (better quality, much slower) — the server
logs which `performance` preset it picked (`auto` by default).

## Security

A LAN-only tool with no authentication. The NSFW safety checker is disabled by
default (`--disable-safety-checker` in all launchers) — do NOT expose the port
to the internet; bind Tailscale (`100.64.0.0/10`) or your LAN subnet only.
