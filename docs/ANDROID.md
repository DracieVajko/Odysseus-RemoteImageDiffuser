# Android — mobil ako server pre Odysseus

Detailná kostra v `android/README.md`, curl checklist v `android/contract-compat.md`.
Tu len PC↔mobil wiring a research verdikty (12 GB RAM, SD 8 Gen2+).

## Režimy appky

- **Local (default):** Ktor :8101 na mobile, whisper.cpp small Q5_0 + LCM-Dreamshaper INT8 (NCNN Vulkan) + Real-ESRGAN-x4v3. Steps napevno 4–8.
- **Remote:** proxy na PC `http://<pc>:8101` (LAN/Tailscale). Vhodné pre SDXL / veľké canvas.
- **Auto:** skúsi Local, pri 503 will-not-load presmeruje na Remote.

## Odysseus napojenie na mobil

1. Mobil + PC na rovnakej LAN / Tailscale.
2. V mobile zapnúť ForegroundService (drží server pri živote).
3. V Odysseus nastaviť image/audio base URL na `http://<mobil-ip>:8101`.
4. Overiť `GET /health` → `status: ok`, `GET /v1/models` → zoznam.

## RAM budget (nesmie cez ~3-4 GB inak OOM killer)

- LCM-Dreamshaper INT8 ~2.5 GB + Whisper small ~350 MB + ESRGAN ~1 GB = sekvenčne, nikdy naraz.
- Preto aj mobil má `ModelManager` single-resident gate (zrkadlo `unified/model_manager.py`).
- PaliGemma-3B (~3 GB) + SD spolu padnú → VLM až V2 a len SmolVLM-500M / Moondream2 int4.

## Licencie (kritické)

- SD-Turbo: non-commercial — pre vlastné použitie OK, redistribúcia NIE.
- LCM-Dreamshaper: OpenRAIL — hlavný kandidát.
- Real-ESRGAN BSD-3, Waifu2x MIT — OK. ffmpeg-kit **len Min/HTTPS (LGPL)** — Full/Main = GPL nákaza.
- Modely sa sťahujú lazy cez ModelDownloader (SHA-256 + .READY), nikdy v APK.
