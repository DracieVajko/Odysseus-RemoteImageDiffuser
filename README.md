# DiffusionServer

Lokálny OpenAI-kompatibilný image server (Stable Diffusion) pre Windows s NVIDIA GPU.
Primárne stavaný ako image backend pre [Odysseus](https://github.com/) — generovanie
mimo Docker hosta, cez Tailscale LAN. Funguje aj samostatne (WebUI, CLI).

- Lazy loading: pri štarte sa nenačíta nič, VRAM ostáva voľná; model sa loadne
  až prvým requestom a ostáva rezidentný (prepnutie = unload + load)
- Modely v `models/` (17–19 GB sa do gitu **nedávajú** — sťahujú sa skriptom)
- `GET /health`, `GET /v1/models`, `POST /v1/images/generations|edits|inpaint|harmonize`
- WebUI na `/webui`, CLI `generate_image.py`, `doctor` diagnostika

## Požiadavky

- Windows 10/11, NVIDIA GPU (odskúšané na 6 GB; 8 GB+ lepšie pre SDXL)
- Python 3.12, ~25 GB voľného miesta (venv ~8 GB + modely ~19 GB)
- Tailscale (voliteľné, na vzdialený prístup)

## Quickstart (5 minút)

```powershell
cd D:\AI\DiffusionServer
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch==2.12.1+cu130 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
.venv\Scripts\python.exe download_models.py --preset odysseus-default
.venv\Scripts\python.exe generate_image.py doctor
```

Štart (lazy, port 8101, aj cez Tailscale):

```powershell
.\Start-Diffusion.ps1 -Lazy -Remote
```

Overenie: http://127.0.0.1:8101/webui — zadaj prompt, generuj.

Plná inštalácia ako služba + firewall + Odysseus prepojenie: viz **INSTALL.md**
(wrátane Linux/systemd sekcie).
Prevádzka, logy, reštarty: **OPERATIONS.md**. Keď niečo padá: **TROUBLESHOOTING.md**.
API kontrakt pre integrácie: **docs/ODYSSEUS-CONTRACT.md**.

## Modely

| ID | Typ | Veľkosť | Zdroj |
|---|---|---|---|
| DreamShaper_8 | SD1.5 | 2.7 GB | `Lykon/dreamshaper-8` (HF, fp16 repo) |
| AbsoluteReality_1.8 | SD1.5 fotorealizmus | 2.0 GB | HF `Lykon/AbsoluteReality` (single-file) |
| AnimagineXL_3.1 | SDXL anime | 6.5 GB | HF `cagliostrolab/...` (single-file) |
| JuggernautXL_v9 | SDXL fotorealizmus | 6.6 GB | HF `RunDiffusion/...` (single-file) |

Vlastný `.safetensors` hoď do `models/<Meno>/` — server ho po reštarte
autodetekuje s hádaným profilom (`guessed:true`), trvalý zápis patrí do
`models.json` (formát viz OPERATIONS.md). Licencie modelov viz karty repozitárov
na HuggingFace (váhy sa sťahujú, nie sú súčasťou repa).

## Výkon (6 GB karta, orientačne)

Cold load: SD1.5 ~1–3 min, SDXL ~3–6 min. Warm: 512px/low rádovo sekundy,
1024px/high minúty. Odporúčané: 768px + medium. Pri pretečení VRAM máš dve
možnosti: (a) menší obrázok/menej krokov len vo VRAM (rýchlejšie, slabšie),
alebo (b) CPU offload so spillom do RAM (lepšia kvalita, oveľa pomalšie) —
server pri štarte zaloguje, ktorý profil zvolil (`performance: auto`).

## Bezpečnosť

LAN-only nástroj bez autentifikácie. Safety checker je defaultne vypnutý
(`--disable-safety-checker` vo všetkých launcheroch) — nezverejňuj port
na internet, len Tailscale (`100.64.0.0/10`).
