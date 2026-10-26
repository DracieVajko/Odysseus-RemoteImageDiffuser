# Odysseus Diffusion Server - RPi Deployment Checklist

## ✅ Windows Side - COMPLETE

### Server Configuration
- **Port**: 8101 (8100 is taken by chromadb)
- **Models**: 4 configured and tested
  - DreamShaper_8 (SD1.5, 768x768, no CPU offload)
  - AbsoluteReality_1.8 (SD1.5, 768x768, no CPU offload)
  - AnimagineXL_3.1 (SDXL, 1024x1024, CPU offload ON)
  - JuggernautXL_v9 (SDXL, 1024x1024, CPU offload ON)
- **Server Status**: Running on 0.0.0.0:8101
- **Models endpoint**: Returns 4 models correctly
- **Health endpoint**: Works, returns `model: null` when idle

### Firewall Status
- ⚠️ **NEEDS FIX**: Run `fix_firewall_8101.ps1` as Administrator
- Current rule exists but missing LocalPort/RemoteAddress/Protocol

---

## 🐳 RPi SIDE - ACTION REQUIRED

### 1. Update Odysseus Endpoint
In Odysseus UI → Settings → AI Defaults → Image Endpoints:
- **OLD**: `http://<WINDOWS-TAILSCALE-IP>:8100/v1`
- **NEW**: `http://<WINDOWS-TAILSCALE-IP>:8101/v1`

### 2. Re-probe Models
In Odysseus UI → Settings → AI Defaults:
- Click "Re-probe" / "Refresh" on the image endpoint
- Should discover 4 models: DreamShaper_8, AbsoluteReality_1.8, AnimagineXL_3.1, JuggernautXL_v9

### 3. Verify Image Generation Dropdown
Settings → AI Defaults → Image Generation:
- Should show all 4 models (not just "Auto-detect" + hardcoded SD models)
- Test "Auto-detect" + each explicit model

### 4. Verify Inpaint Still Works
Settings → AI Defaults → Inpaint:
- Should show all 4 models
- Test inpaint with each model

---

## 📋 VERIFICATION COMMANDS

### From RPi (or any Tailscale device):
```bash
# Health check
curl http://<WINDOWS-TAILSCALE-IP>:8101/health

# Models list
curl http://<WINDOWS-TAILSCALE-IP>:8101/v1/models

# Test generation
curl -X POST http://<WINDOWS-TAILSCALE-IP>:8101/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"model":"DreamShaper_8","prompt":"test","size":"512x512"}'

# Test inpaint
curl -X POST http://<WINDOWS-TAILSCALE-IP>:8101/v1/images/inpaint \
  -H "Content-Type: application/json" \
  -d '{"model":"DreamShaper_8","image":"...","mask":"...","prompt":"test","width":512,"height":512}'
```

---

## 📊 Performance Reference (RTX 3060 6GB)

| Model | Type | Resolution | Cold Load | Warm Gen | VRAM |
|-------|------|------------|-----------|----------|------|
| DreamShaper_8 | SD1.5 | 768x768 / 1024x1024 | ~85s / 13s | ~13s | ~3.1 GB |
| AbsoluteReality_1.8 | SD1.5 | 768x768 / 1024x1024 | ~46s / 14s | ~14s | ~3.1 GB |
| AnimagineXL_3.1 | SDXL | 1024x1024 / 1920x1080 | ~179s / 30s | ~30s | ~5.5 GB |
| JuggernautXL_v9 | SDXL | 1024x1024 / 1920x1080 | ~199s / 38s | ~38s | ~6.5 GB |

*SD1.5 models: Direct GPU (no CPU offload, fast)*
*SDXL models: CPU offload ON (fits in 6GB VRAM, slower)*

---

## 🔧 Server Management

```powershell
# Start server (manual)
cd D:\AI\DiffusionServer
cmd /c run_diffusion_persistent.bat

# Or use the menu
python run_diffusion_menu.py

# Stop server
.\Stop-Diffusion.ps1

# Check status
.\Test-Diffusion.ps1
```

### Service (auto-start)
```powershell
# Install (already done)
.\Install-DiffusionStartup.ps1 -Remote

# Start now
Start-ScheduledTask -TaskName OdysseusDiffusionHelper

# Stop
.\Stop-Diffusion.ps1
```

---

## 📁 Key Files
| File | Purpose |
|------|---------|
| `diffusion_server.py` | Main server with lazy loading |
| `models.json` | 4 model configs |
| `run_diffusion_menu.py` | Interactive generation menu |
| `run_diffusion_persistent.bat` | Persistent server launcher |
| `fix_firewall_8101.ps1` | **Run as Admin to fix firewall** |
| `Start-Diffusion.ps1` | Server launcher |
| `Stop-Diffusion.ps1` | Clean shutdown |
| `Test-Diffusion.ps1` | Health check |
| `Install-DiffusionStartup.ps1` | Service installer |

---

## 🔗 Endpoints Summary
| Endpoint | Local | Tailscale (RPi) |
|----------|-------|-----------------|
| Health | http://127.0.0.1:8101/health | http://<WINDOWS-TAILSCALE-IP>:8101/health |
| Models | http://127.0.0.1:8101/v1/models | http://<WINDOWS-TAILSCALE-IP>:8101/v1/models |
| Generate | http://127.0.0.1:8101/v1/images/generations | http://<WINDOWS-TAILSCALE-IP>:8101/v1/images/generations |
| Inpaint | http://127.0.0.1:8101/v1/images/inpaint | http://<WINDOWS-TAILSCALE-IP>:8101/v1/images/inpaint |

---

## ⚡ QUICK START
1. **Run as Admin**: `powershell -File D:\AI\DiffusionServer\fix_firewall_8101.ps1`
2. **RPi**: Update endpoint to `http://<WINDOWS-TAILSCALE-IP>:8101/v1`
3. **RPi**: Re-probe in Odysseus UI
4. **Test**: Image Generation dropdown should show all 4 models
5. **Test**: Generate + Inpaint with each model

**Windows server is 100% ready. Only firewall fix + RPi endpoint update needed.**