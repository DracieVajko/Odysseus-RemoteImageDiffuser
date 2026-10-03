# INSTALL (Windows)

## 1. Python a venv

Nainštaluj Python 3.12 (python.org, zaškrtni "Add to PATH"):

```powershell
cd <cesta>\DiffusionServer
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.12.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

## 2. Modely

```powershell
.venv\Scripts\python.exe download_models.py --list        # čo existuje
.venv\Scripts\python.exe download_models.py --preset odysseus-default   # ~19 GB
# len SD1.5 (~5 GB): --preset sd15 | len SDXL (~14 GB): --preset sdxl
# Civitai (treba API token z nastavení účtu):
.venv\Scripts\python.exe download_models.py --civitai <versionId> --target Moja --name model.safetensors --civitai-token <token>
```

## 3. Kontrola

```powershell
.venv\Scripts\python.exe generate_image.py doctor
```

Všetko `[OK]`, `server: UNREACHABLE` je v poriadku (ešte nebeží).

## 4. Služba (autostart po reboote, odporúčané)

**Ako Administrátor** (štart menu → PowerShell → Run as Administrator):

```powershell
cd <cesta>\DiffusionServer
.\Install-DiffusionStartup.ps1 -Remote   # ak existuje helper úloha, alebo:
.\install_service_admin.ps1             # NSSM služba OdysseusDiffusion
.\fix_firewall_8101.ps1                # port 8101 pre Tailscale 100.64.0.0/10
Restart-Service OdysseusDiffusion
.\Test-Diffusion.ps1
```

Pravidlo: **buď služba, alebo naplánovaná úloha — nikdy obe** (kolidujú na porte).
Odporúčaná je služba (štartuje pred prihlásením).

## 5. Manuálny štart (bez admina)

```powershell
.\Start-Diffusion.ps1 -Lazy -Remote   # lazy, 8101, aj Tailscale
.\Stop-Diffusion.ps1
```

## 6. Odysseus prepojenie

V Odysseus `Settings → AI Defaults → Image Endpoints`:
`http://<WINDOWS-TAILSCALE-IP>:8101/v1`, potom re-probe modelov.
Očakávané: 4 modely (`available:true`, `missing_models:[]`).

## 7. Linux (server)

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130  # alebo cpu build
pip install -r requirements.txt
cp models.json.example models.json
python download_models.py --preset odysseus-default
python generate_image.py doctor
sudo cp systemd/diffusion-server.service /etc/systemd/system/
sudo cp systemd/diffusion-server.env.example /etc/diffusion-server.env  # uprav cesty
sudo systemctl enable --now diffusion-server
curl http://127.0.0.1:8101/health
```

Env overrides: `DIFFUSION_PORT`, `DIFFUSION_HOST`, `DIFFUSION_MODELS_JSON`.

### Raspberry Pi 5 / ARM64 (bez overenia)

Neotestované na ARM — podľa špecifikácie očakávaj: preset `cpu` (žiadna CUDA),
minimálne 8 GB RAM, modely SD1.5. Prakticky: `performance: cpu`, 512px + low
a trpezlivosť (CPU generovanie meria v desiatkach minút). Pre použiteľné časy
na CPU zváž SD-Turbo class modely (4 kroky stačia) ako alternatívny profil.
Ak to niekto overí na RPi5, uvítame čísla do tejto sekcie.
