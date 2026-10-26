#!/usr/bin/env python3
"""Benchmark: same prompt on all 4 models, profile size + high quality.

Saves output/<model>.png + output/benchmark.json (timings + image stats).
Long run (SDXL cold loads take minutes) - run in background.
"""
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8101"
OUTDIR = Path(__file__).resolve().parent / "output"
PROMPT = "cute girl long hair japanese face black lips blue eyes"

# (model, size): profile defaults (SD1.5 768, SDXL 1024), high quality
JOBS = [
    ("DreamShaper_8", "768x768"),
    ("AbsoluteReality_1.8", "768x768"),
    ("AnimagineXL_3.1", "1024x1024"),
    ("JuggernautXL_v9", "1024x1024"),
]


def stats(path):
    from PIL import Image
    import numpy as np
    arr = np.asarray(Image.open(path).convert("RGB")).astype(float)
    return {"mean": round(float(arr.mean()), 1), "std": round(float(arr.std()), 1),
            "black_pct": round(float((arr.sum(axis=2) == 0).mean() * 100), 1)}


def main():
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    OUTDIR.mkdir(exist_ok=True)
    results = {}
    for model, size in JOBS:
        if only and model not in only:
            continue
        rid = f"bench-{model}"
        payload = {"model": model, "prompt": PROMPT, "size": size,
                   "quality": "high", "n": 1, "request_id": rid}
        req = urllib.request.Request(BASE + "/v1/images/generations",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"},
                                     method="POST")
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                body = json.loads(r.read().decode())
            elapsed = time.time() - t0
            raw = base64.b64decode(body["data"][0]["b64_json"])
            dest = OUTDIR / f"{model}.png"
            dest.write_bytes(raw)
            st = stats(dest)
            results[model] = {"status": 200, "elapsed_s": round(elapsed, 1),
                              "bytes": len(raw), **st}
            print(f"[{model}] 200 in {elapsed:.0f}s mean={st['mean']} "
                  f"std={st['std']} black={st['black_pct']}% -> {dest.name}", flush=True)
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode()[:300]
            except Exception:
                detail = ""
            results[model] = {"status": e.code, "elapsed_s": round(time.time() - t0, 1),
                              "error": detail}
            print(f"[{model}] {e.code} after {time.time()-t0:.0f}s: {detail}", flush=True)
        except Exception as e:  # noqa: BLE001
            results[model] = {"status": "exception", "error": str(e)[:200]}
            print(f"[{model}] EXC: {e}", flush=True)
    (OUTDIR / "benchmark.json").write_text(json.dumps(
        {"prompt": PROMPT, "quality": "high", "results": results}, indent=1))
    print("SAVED output/benchmark.json", flush=True)


if __name__ == "__main__":
    main()
