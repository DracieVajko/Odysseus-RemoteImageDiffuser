#!/usr/bin/env python3
"""Reproduce the reported black-PNG case + objectively measure blackness.

Prompt/quality exactly as reported: 'cute girl long hair', DreamShaper_8, high.
Prints mean/std/black-pixel ratio so we can tell filter-black (pure #000)
from VAE/grey failure modes.
"""
import base64
import json
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8101"
BASE = f"http://127.0.0.1:{PORT}"
OUT = rf"D:\AI\DiffusionServer\safetytest-{PORT}.png"


def main():
    payload = {"model": "DreamShaper_8", "prompt": "cute girl long hair",
               "size": "512x512", "quality": "high", "n": 1}
    req = urllib.request.Request(BASE + "/v1/images/generations",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read().decode())
    print(f"elapsed={time.time()-t0:.1f}s", flush=True)
    raw = base64.b64decode(body["data"][0]["b64_json"])
    with open(OUT, "wb") as f:
        f.write(raw)
    from PIL import Image
    import numpy as np
    arr = np.asarray(Image.open(OUT).convert("RGB")).astype(float)
    black = (arr.sum(axis=2) == 0).mean()
    print(f"bytes={len(raw)} mean={arr.mean():.1f} std={arr.std():.1f} "
          f"pure_black_px={black*100:.1f}% saved->{OUT}", flush=True)


if __name__ == "__main__":
    main()
