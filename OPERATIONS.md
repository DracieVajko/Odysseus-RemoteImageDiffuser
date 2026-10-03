# OPERATIONS

## Service

```powershell
Get-Service OdysseusDiffusion
Start-Service OdysseusDiffusion
Restart-Service OdysseusDiffusion
Stop-Service OdysseusDiffusion
.\Test-Diffusion.ps1            # health + models + VRAM + log tail
```

After a restart the server is lazy: `model: null`. The first request loads a
model (SD1.5 ~1–3 min, SDXL ~3–6 min). Switching models = unload + load; while
loading, requests get `503 {"error":"loading model"}`. `/health` shows
`busy` / `queue` / `current_model` so clients can display "server is working".

## Logs

- `logs/service-out.log` — uvicorn access log (NSSM stdout)
- `logs/service-err.log` — app log: loading, timings
  (`Generated/Edited/Inpaint done in Xs`), `Safety checker BLACKED`,
  `Model ready`, hardware preset, auto-discovery WARNINGs
- Every generation log line carries `[rid=xxxxxxxx]` for tracing
- Manual runs: `--log-file name.log --log-level DEBUG`

## models.json

Copy `models.json.example` → `models.json` and adjust paths:

```json
{
  "port": 8101,
  "auto_discover": true,
  "performance": "auto",
  "default_model": "DreamShaper_8",
  "models": {
    "MyIdentity": {
      "label": "Label for UI",
      "path": "models/My/model.safetensors",
      "single_file_config": "author/repo-with-configs",
      "pipeline": "StableDiffusionXLPipeline",
      "dtype": "float16", "steps": 25, "guidance_scale": 7.0,
      "width": 1024, "height": 1024,
      "attention_slicing": true, "vae_slicing": true, "cpu_offload": true
    }
  }
}
```

- `path` = file (single-file checkpoint) or directory (diffusers format,
  relative to server dir or absolute)
- `single_file_config` = HuggingFace repo with configs for single files
  (use an open one, e.g. `stable-diffusion-v1-5/stable-diffusion-v1-5`)
- `performance`: `auto` (from VRAM) or `low|mid|high|cpu`
- After config changes: restart + `generate_image.py doctor`
- Unknown files under `models/` are proposed automatically after restart
  (`guessed:true` in `/v1/models`) — confirm the profile in the config

## Quality → steps

low=4, medium=profile default (25), high=30, auto=12. Use medium for tests.

## CLI

```powershell
python generate_image.py health | models | doctor
python generate_image.py txt2img --model DreamShaper_8 --prompt "..." --output out.png
python generate_image.py img2img --model ... --prompt "..." --input ref.png --output out.png --strength 0.75
python generate_image.py inpaint --model ... --prompt "..." --input img.png --mask mask.png --output out.png
```

## Backup

`models.json` + this repo is enough. Weights can always be re-downloaded
(`download_models.py --preset odysseus-default`).
