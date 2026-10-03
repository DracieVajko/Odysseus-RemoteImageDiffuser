# OPERATIONS

## Služba

```powershell
Get-Service OdysseusDiffusion
Start-Service OdysseusDiffusion
Restart-Service OdysseusDiffusion
Stop-Service OdysseusDiffusion
.\Test-Diffusion.ps1            # health + modely + VRAM + chvost logu
```

Po reštarte je server lazy: `model: null`, prvý request model načíta (SD1.5
~1–3 min, SDXL ~3–6 min). Prepnutie modelu = unload + load, medzitým requesty
dostanú `503 {"error":"loading model"}`.

## Logy

- `logs/service-out.log` — uvicorn access log (NSSM stdout)
- `logs/service-err.log` — aplikačný log: loadovanie, časovania
  (`Generated/Edited/Inpaint done in Xs`), `Safety checker BLACKED`,
  `Model ready`, hardware preset, auto-discovery WARNINGy
- Každý request má `[rid=xxxxxxxx]` v log riadkoch
- Manuálny beh: `--log-file meno.log --log-level DEBUG`

## models.json

```json
{
  "port": 8101,
  "auto_discover": true,
  "performance": "auto",
  "default_model": "DreamShaper_8",
  "models": {
    "MojaIdentita": {
      "label": "Popis do UI",
      "path": "D:\\AI\\DiffusionServer\\models\\Moja\\model.safetensors",
      "single_file_config": "autor/repo-s-configmi",
      "pipeline": "StableDiffusionXLPipeline",
      "dtype": "float16", "steps": 25, "guidance_scale": 7.0,
      "width": 1024, "height": 1024,
      "attention_slicing": true, "vae_slicing": true, "cpu_offload": true
    }
  }
}
```

- `path` = súbor (single-file) alebo adresár (diffusers formát)
- `single_file_config` = HF repo s configmi pre single-file (povinné u nich)
- `performance`: `auto` (podľa VRAM) alebo `low|mid|high|cpu`
- Po zmene configu: `Restart-Service` + `generate_image.py doctor`
- Neznáme súbory v `models/` server po reštarte navrhne sám (`guessed:true`
  v `/v1/models`) — profil potvrď v configu

## Kvalita → kroky

low=4, medium=profilový default (25), high=30, auto=12. Pre testy medium.

## Záloha

Stačí `models.json` + tento repozitár. Váhy sa dajú vždy stiahnuť znova
(`download_models.py --preset odysseus-default`).
