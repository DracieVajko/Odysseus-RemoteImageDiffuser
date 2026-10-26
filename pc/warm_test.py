#!/usr/bin/env python3
"""Warm test: cold-load DreamShaper_8 and verify a real PNG comes back."""
import base64
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8101"
OUT = r"D:\AI\DiffusionServer\warmtest.png"

if len(sys.argv) > 1:
    BASE = f"http://127.0.0.1:{sys.argv[1]}"
    OUT = rf"D:\AI\DiffusionServer\warmtest-{sys.argv[1]}.png"

MODEL = sys.argv[2] if len(sys.argv) > 2 else "DreamShaper_8"
if len(sys.argv) > 2:
    OUT = OUT.replace(".png", f"-{MODEL}.png")


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.status, json.loads(r.read().decode())


def main():
    t0 = time.time()
    status, body = post("/v1/images/generations", {
        "model": MODEL,
        "prompt": "a cozy mountain cabin at sunset, pine forest",
        "size": "512x512", "quality": "low", "n": 1,
    })
    elapsed = time.time() - t0
    data = body.get("data", [])
    print(f"status={status} elapsed={elapsed:.1f}s images={len(data)}", flush=True)
    if not data:
        print("BODY:", json.dumps(body)[:500], flush=True)
        sys.exit(1)
    raw = base64.b64decode(data[0]["b64_json"])
    print(f"bytes={len(raw)} png_header={raw[:8].hex()} is_png={raw[:8] == bytes.fromhex('89504e470d0a1a0a')}", flush=True)
    with open(OUT, "wb") as f:
        f.write(raw)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
