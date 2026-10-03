# INSTALL

## Windows

### 1. Python and venv

Install Python 3.12 (python.org, check "Add to PATH"):

```powershell
cd <path>\Odysseus-RemoteImageDiffuser
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.12.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### 2. Models

```powershell
.venv\Scripts\python.exe download_models.py --list        # what's available
.venv\Scripts\python.exe download_models.py --preset odysseus-default   # ~19 GB
# SD1.5 only (~5 GB): --preset sd15 | SDXL only (~14 GB): --preset sdxl
# Single model:  python download_models.py AbsoluteReality_1.8
# Civitai (needs an API token from account settings):
python download_models.py --civitai <versionId> --target MyModel --name model.safetensors --civitai-token <token>
# (find the version ID in the model page URL: ...?modelVersionId=<ID>)
```

### 3. Sanity check

```powershell
.venv\Scripts\python.exe generate_image.py doctor
```

Everything `[OK]`; `server: UNREACHABLE` is fine (not running yet).

### 4. Service (autostart after reboot, recommended)

**As Administrator** (Start menu → PowerShell → Run as Administrator):

```powershell
cd <path>\Odysseus-RemoteImageDiffuser
.\install_service_admin.ps1            # NSSM service, auto-detects Tailscale IP
.\fix_firewall_8101.ps1               # opens port 8101 for Tailscale
Restart-Service OdysseusDiffusion
.\Test-Diffusion.ps1
```

Rule: **service OR scheduled task — never both** (they fight over the port).
The service is recommended (starts before login).

### 5. Manual start (no admin)

```powershell
.\Start-Diffusion.ps1 -Lazy -Remote   # lazy, 8101, Tailscale-friendly
.\Stop-Diffusion.ps1
# or: run_diffusion_port8101.bat      # portable, no admin, honors DIFFUSION_PORT/HOST
```

### 6. Remote access: Tailscale vs plain LAN

Both work — pick one:

**A. Tailscale (recommended for Odysseus across machines):**
`Start-Diffusion.ps1 -Remote` (or the service) binds `0.0.0.0` and allows your
Tailscale IP via `--allowed-host` (auto-detected). Firewall:
`fix_firewall_8101.ps1` (scope `100.64.0.0/10`). URL from another machine:
`http://<WINDOWS-TAILSCALE-IP>:8101`.

**B. Plain internal LAN (e.g. `192.168.1.50`):**
pass the LAN IP explicitly and open the firewall for your subnet:

```powershell
# service install with LAN IP:
.\install_service_admin.ps1 -TailscaleIp 192.168.1.50
# firewall for LAN only (admin):
New-NetFirewallRule -DisplayName "Diffusion 8101 (LAN)" -Direction Inbound `
  -Protocol TCP -LocalPort 8101 -RemoteAddress 192.168.1.0/24 -Action Allow
```

URL from another machine: `http://192.168.1.50:8101`. Never expose 8101 to
the internet (no auth on the server).

## Linux (server)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130  # or the CPU build
pip install -r requirements.txt
cp models.json.example models.json
python download_models.py --preset odysseus-default
python generate_image.py doctor
sudo cp systemd/diffusion-server.service /etc/systemd/system/
sudo cp systemd/diffusion-server.env.example /etc/diffusion-server.env  # edit paths
sudo systemctl enable --now diffusion-server
curl http://127.0.0.1:8101/health
```

Env overrides: `DIFFUSION_PORT`, `DIFFUSION_HOST`, `DIFFUSION_MODELS_JSON`.

### Raspberry Pi 5 / ARM64 (not verified)

Untested on ARM — expect: `cpu` preset (no CUDA), 8 GB+ RAM, SD1.5 models.
Practical setup: `performance: cpu`, 512px + low quality, and patience
(CPU generation takes tens of minutes). For usable CPU times consider
SD-Turbo-class models (4 steps are enough) as an alternative profile.
If you verify numbers on RPi5, contributions to this section are welcome.

## Wiring into Odysseus

1. In Odysseus go to `Settings → AI Defaults → Image Endpoints` and add:
   `http://<SERVER-IP>:8101/v1` (Tailscale IP or LAN IP of this machine —
   note the `/v1` suffix).
2. Re-probe / refresh models. Expected: 4 models, all `available:true`,
   `missing_models:[]`.
3. Generate once per model (cold load takes minutes, later requests are fast).
   If a request outlives the Odysseus timeout, the server still finishes it —
   raise the client timeout (300–420 s for 1024px/high) or use 768px + medium.
4. Contract details (timeouts, progress polling, status codes):
   `docs/ODYSSEUS-CONTRACT.md`.
