#!/usr/bin/env python3
"""Download diffusion models into DiffusionServer/models/ (resumable).

Presets (all from public HuggingFace repos, no login needed):
  odysseus-default : all 4 models below (~19 GB)
  sd15             : DreamShaper_8 + RealisticVision (~5 GB)
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
]

PRESETS = {
    "odysseus-default": [t["id"] for t in TASKS],
    "sd15": ["DreamShaper_8", "AbsoluteReality_1.8"],
    "sdxl": ["AnimagineXL_3.1", "JuggernautXL_v9"],
}

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
            p = hf_hub_download(task["repo"], task["file"], local_dir=str(target))
            got, dest = Path(p), target / task["file"]
            if got != dest:
                if dest.exists():
                    dest.unlink()
                got.rename(dest)
            return dest

        run_with_retries(_dl, task["id"])
    else:
        print(f"[{task['id']}] snapshot {task['repo']} -> {target} ...", flush=True)
        run_with_retries(
            lambda: snapshot_download(task["repo"], local_dir=str(target),
                                      allow_patterns=task["allow"]), task["id"])
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
        for t in TASKS:
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
    unknown = [i for i in ids if i not in {t["id"] for t in TASKS}]
    if unknown:
        print(f"Unknown model IDs: {', '.join(unknown)}", flush=True)
        sys.exit(2)
    BASE.mkdir(exist_ok=True)
    for t in TASKS:
        if t["id"] not in ids:
            continue
        try:
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
