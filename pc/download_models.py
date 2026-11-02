#!/usr/bin/env python3
"""Download diffusion models into DiffusionServer/models/ (resumable).

Presets (all from public HuggingFace repos, no login needed):
  odysseus-default : all 4 models below (~19 GB)
  sd15             : DreamShaper_8 + AbsoluteReality_1.8 (~5 GB)
  sdxl             : AnimagineXL_3.1 + JuggernautXL_v9 (~14 GB)

Usage:
  python download_models.py --list
  python download_models.py --preset odysseus-default
  python download_models.py DreamShaper_8 JuggernautXL_v9
  python download_models.py --civitai 12345 --target models/MyModel --name model.safetensors --civitai-token ABC
"""
import argparse
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent / "models"

# Cancellation flags (model_id -> set by models_manager.cancel)
_CANCEL: set = set()


class CancelledError(Exception):
    pass


def _check_cancel(task_id: str):
    if task_id in _CANCEL:
        raise CancelledError(f"cancelled: {task_id}")

TASKS = [
    {"id": "DreamShaper_8", "kind": "snapshot", "repo": "Lykon/dreamshaper-8",
     "target": "DreamShaper_8", "min_gb": 2.0,
     "allow": ["*.fp16.safetensors", "*.json", "*.txt", "*.model",
               "tokenizer/*", "scheduler/*", "feature_extractor/*"],
     "note": "official diffusers repo, fp16 weights + configs (no single-file on HF)",
     "license": "CreativeML Open RAIL-M (see repo card)"},
    {"id": "AbsoluteReality_1.8", "kind": "file",
     "repo": "Lykon/AbsoluteReality",
     "file": "AbsoluteReality_1.8.1_pruned.safetensors",
     "target": "AbsoluteReality_1.8", "min_gb": 1.5,
     "license": "CreativeML Open RAIL-M (see repo card)"},
    {"id": "AnimagineXL_3.1", "kind": "file", "repo": "cagliostrolab/animagine-xl-3.1",
     "file": "animagine-xl-3.1.safetensors", "target": "AnimagineXL_3.1", "min_gb": 5.5,
     "license": "Fair AI Public License 1.0-SD (see repo card)"},
    {"id": "JuggernautXL_v9", "kind": "file", "repo": "RunDiffusion/Juggernaut-XL-v9",
     "file": "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
     "target": "JuggernautXL_v9", "min_gb": 5.5,
     "license": "CreativeML Open RAIL-M (see repo card)"},
    {"id": "SDXL_Turbo", "kind": "file", "repo": "stabilityai/sdxl-turbo",
     "file": "sd_xl_turbo_1.0_fp16.safetensors",
     "target": "SDXL_Turbo", "min_gb": 6.0,
     "single_file_config": "stabilityai/sdxl-turbo",
     "pipeline": "StableDiffusionXLPipeline",
     "license": "OpenRAIL++-M (see repo card)"},
]

PRESETS = {
    "odysseus-default": [t["id"] for t in TASKS],
    "sd15": ["DreamShaper_8", "AbsoluteReality_1.8"],
    "sdxl": ["AnimagineXL_3.1", "JuggernautXL_v9"],
    "sdxl-turbo": ["SDXL_Turbo"],
    "upscale": ["RealESRGAN_x4v3"],
    "vlm": ["SmolVLM_500M"],
}

# VLM (V2): HuggingFaceTB SmolVLM-500M-Instruct (~1 GB, Apache-2.0).
# Snapshot like SD models; the server uses transformers AutoModelForImageTextToText.
VLM_TASKS = [
    {"id": "SmolVLM_500M", "kind": "snapshot", "repo": "HuggingFaceTB/SmolVLM-500M-Instruct",
     "target": "SmolVLM_500M", "min_gb": 0.8,
     "allow": ["*.safetensors", "*.json", "tokenizer/*", "preprocessor_config.json",
               "*.model", "*.txt", "chat_template.jinja"],
     "license": "Apache-2.0 (see repo card)"},
]

# Upscale weights (small, also downloaded to the mobile package via ModelDownloader).
# RealESRGAN_x4plus.pth: x4, general photos (BSD-3-Clause, xinntao/Real-ESRGAN,
# stable GitHub release asset — no official HF repo for this weight).
UPSCALE_TASKS = [
    {"id": "RealESRGAN_x4v3", "kind": "url",
     "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
     "file": "RealESRGAN_x4plus.pth", "target": "RealESRGAN_x4v3",
     "min_gb": 0.05,
     "license": "BSD-3-Clause (see repo card)"},
]

FAILED = []


def run_with_retries(fn, label, tries=5):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - network flakes, retry
            print(f"[{label}] attempt {attempt}/{tries} failed: {str(e)[:200]}", flush=True)
            if attempt == tries:
                raise
            time.sleep(10 * attempt)


def dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def download_url(task):
    import requests
    target = BASE / task["target"]
    target.mkdir(parents=True, exist_ok=True)
    dest = target / task["file"]
    if dest.exists() and dest.stat().st_size / 1e9 >= task["min_gb"]:
        print(f"[{task['id']}] exists, skipping ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
        return
    print(f"[{task['id']}] downloading {task['url']} ...", flush=True)

    def _dl():
        with requests.get(task["url"], stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done = 0
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    _check_cancel(task["id"])
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
            tmp.rename(dest)
        print(f"\n[{task['id']}] DONE ({dest.stat().st_size / 1e6:.1f} MB in {dest})", flush=True)

    run_with_retries(_dl, task["id"])


def download_hf(task):
    from huggingface_hub import hf_hub_download, snapshot_download
    target = BASE / task["target"]
    target.mkdir(parents=True, exist_ok=True)
    if dir_size_gb(target) >= task["min_gb"]:
        print(f"[{task['id']}] exists, skipping ({dir_size_gb(target):.2f} GB)", flush=True)
        return
    if task["kind"] == "file":
        print(f"[{task['id']}] downloading {task['file']} ...", flush=True)

        def _dl():
            p = hf_hub_download(task["repo"], task["file"], local_dir=str(target), revision="main")
            got, dest = Path(p), target / task["file"]
            if got != dest:
                if dest.exists():
                    dest.unlink()
                got.rename(dest)
            return dest

        run_with_retries(_dl, task["id"])
    else:
        import fnmatch
        from huggingface_hub import list_repo_files
        print(f"[{task['id']}] snapshot {task['repo']} -> {target} ...", flush=True)
        allow = task.get("allow") or ["*"]
        all_files = run_with_retries(lambda: list_repo_files(task["repo"]), task["id"])
        files = [f for f in all_files if any(fnmatch.fnmatch(f, pat) for pat in allow)]
        print(f"[{task['id']}] {len(files)} files", flush=True)
        for rel in files:
            _check_cancel(task["id"])
            run_with_retries(
                lambda rel=rel: hf_hub_download(task["repo"], rel, local_dir=str(target), revision="main"),
                task["id"],
            )
    size = dir_size_gb(target)
    if size < task["min_gb"]:
        raise RuntimeError(f"downloaded only {size:.2f} GB, expected >= {task['min_gb']} GB")
    print(f"[{task['id']}] DONE ({size:.2f} GB in {target})", flush=True)


def download_civitai(version_id: str, target_dir: str, filename: str, token: str):
    """Download a Civitai model version by numeric version ID.

    Find the ID on the model page URL (.../models/<modelId>?modelVersionId=<ID>).
    Token: Civitai account Settings -> API keys (required by Civitai for downloads).
    """
    import requests
    target = BASE / target_dir
    target.mkdir(parents=True, exist_ok=True)
    dest = target / filename
    if dest.exists() and dest.stat().st_size > 100 * 1024 * 1024:
        print(f"[civitai:{version_id}] exists, skipping ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)
        return
    url = f"https://civitai.com/api/download/models/{version_id}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    print(f"[civitai:{version_id}] downloading -> {dest} ...", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print(f"\n[civitai:{version_id}] DONE ({dest.stat().st_size / 1e9:.2f} GB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Download models into DiffusionServer/models/")
    ap.add_argument("models", nargs="*", help="model IDs (default: preset)")
    ap.add_argument("--preset", default="odysseus-default", choices=list(PRESETS),
                    help="download preset")
    ap.add_argument("--list", action="store_true", help="list models/presets and exit")
    ap.add_argument("--civitai", default="", help="Civitai model VERSION id to download")
    ap.add_argument("--target", default="", help="target subdir under models/ (civitai)")
    ap.add_argument("--name", default="", help="output filename (civitai)")
    ap.add_argument("--civitai-token", default=os.environ.get("CIVITAI_TOKEN", ""),
                    help="Civitai API token (or CIVITAI_TOKEN env)")
    args = ap.parse_args()

    if args.list:
        print("Models:", flush=True)
        for t in TASKS + UPSCALE_TASKS + VLM_TASKS:
            print(f"  {t['id']}: {t.get('repo', 'civitai')} [{t.get('license', '')}]", flush=True)
        print("Presets:", flush=True)
        for name, ids in PRESETS.items():
            print(f"  {name}: {', '.join(ids)}", flush=True)
        return

    if args.civitai:
        if not args.target or not args.name:
            ap.error("--civitai needs --target and --name")
        try:
            download_civitai(args.civitai, args.target, args.name, args.civitai_token)
        except Exception as e:  # noqa: BLE001
            print(f"[civitai:{args.civitai}] FAILED: {e}", flush=True)
            sys.exit(1)
        print("ALL DONE", flush=True)
        return

    ids = args.models or PRESETS[args.preset]
    all_tasks = TASKS + UPSCALE_TASKS + VLM_TASKS
    unknown = [i for i in ids if i not in {t["id"] for t in all_tasks}]
    if unknown:
        print(f"Unknown model IDs: {', '.join(unknown)}", flush=True)
        sys.exit(2)
    BASE.mkdir(exist_ok=True)
    for t in all_tasks:
        if t["id"] not in ids:
            continue
        try:
            if t.get("kind") == "url":
                download_url(t)
            else:
                download_hf(t)
        except Exception as e:  # noqa: BLE001
            print(f"[{t['id']}] FAILED: {e}", flush=True)
            FAILED.append(t["id"])
    if FAILED:
        print("FAILED:", ", ".join(FAILED), flush=True)
        sys.exit(1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
