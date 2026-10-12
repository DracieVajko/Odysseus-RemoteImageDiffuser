# Changelog

## 0.5.0 (unreleased)
- VRAM-aware loading: fits VRAM → CUDA; VRAM short but RAM fits → CPU fp32
  fallback (slow but works); fits nowhere → immediate 503 `will not load`
  (no more hanging into client timeouts). OOM during load → same 503.
- `device` (cuda/cpu) in `/health`; derived pipes follow the active dtype

## 0.4.0 (unreleased)
- Everything in English: docs, WebUI, scripts, contract
- LAN setup alongside Tailscale (`-TailscaleIp` = any IP, LAN firewall example)
- Detailed Odysseus wiring guide (endpoint + re-probe + timeouts)
- Model swap: RealisticVision out, AbsoluteReality_1.8 in (all references)
- Portable launchers (relative paths, Tailscale auto-detect, env overrides)

## 0.3.0 (unreleased)
- Public-repo release: `models.json.example`, `systemd/` unit + env example,
  `LICENSE` (MIT), Linux quickstart (incl. honest RPi5/ARM64 note)
- Secrets sweep: no IPs/paths in launchers (Tailscale auto-detect, `$ServerDir`
  relative paths, `DIFFUSION_PORT/HOST/MODELS_JSON` env overrides)
- `install_service_admin.ps1`: `-TailscaleIp` param, portable paths

## 0.2.0 (unreleased)
- Release package: README, INSTALL, OPERATIONS, TROUBLESHOOTING,
  `docs/ODYSSEUS-CONTRACT.md`, `requirements.txt` (pinned), `.gitignore`
- CLI: `generate_image.py` subcommands (`txt2img|img2img|inpaint|models|doctor|health`)
- `download_models.py`: presets (`odysseus-default|sd15|sdxl`), `--list`,
  Civitai `--civitai` download with token, size verification
- Model auto-discovery (`auto_discover`, `guessed` flag) + `performance`
  presets (`auto|low|mid|high|cpu`, VRAM+RAM detection, low-VRAM offload rule)
- `--log-file` / `--log-level`, `request_id` in log lines, WebUI: strength
  slider, busy/queue status, settings in localStorage, refresh button
- `high` quality mapping fixed to 30 steps (was 20, less than medium)
- install_service_admin: NSSM env fixed (per-VAR entries, was one `;` blob)

## 0.1.0 (unreleased)
- Lazy-loading OpenAI-compatible image server (port 8101): generations, edits
  (native img2img), inpaint, harmonize, progress polling, `/webui`
- Model registry (`models.json`) + `available` / `missing_models` reporting
- `--disable-safety-checker`, fp16-only repo support (`variant="fp16"`)
- Cold-load 503 + `busy` / `queue` in `/health`, 400 validation on bad inputs
- Windows service (NSSM), firewall + Tailscale setup, test/panel scripts
