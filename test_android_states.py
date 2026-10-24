"""Android /health state test: busy and queue must follow real requests.

Requires the app running with the HTTP server started and adb forward 8199.
"""
import base64
import os
import threading
import time
from pathlib import Path

import httpx

BASE = os.getenv("PRISM_ANDROID", "http://127.0.0.1:8199")
c = httpx.Client(base_url=BASE, timeout=1200.0)
TD = Path(r"D:\AI\DiffusionServer\testdata")
img_b64 = base64.b64encode((TD / "images" / "test_512.png").read_bytes()).decode()
mask_b64 = base64.b64encode((TD / "images" / "mask_512.png").read_bytes()).decode()
failures = []


def health():
    return c.get("/health").json()


def expect(name, cond, detail=""):
    print(f"[{'OK  ' if cond else 'FAIL'}] {name} {detail}", flush=True)
    if not cond:
        failures.append(name)


def wait_for(predicate, timeout=120.0, interval=0.5):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        last = health()
        if predicate(last):
            return last
        time.sleep(interval)
    return last


def post(tag, path, payload, out):
    try:
        r = c.post(path, json=payload)
        out[tag] = r.status_code
    except Exception as e:  # pragma: no cover
        out[tag] = f"EXC {e}"


h = health()
expect("idle at start", h["busy"] is False and h["queue"] == 0,
       f"busy={h['busy']} queue={h['queue']}")

print("\n=== TWO CONCURRENT REQUESTS ===")
out = {}
t1 = threading.Thread(target=post, args=(
    "gen", "/v1/images/generations",
    {"prompt": "a red cube", "size": "384x384", "steps": 6, "n": 1}, out))
t2 = threading.Thread(target=post, args=(
    "inpaint", "/v1/images/inpaint",
    {"image": img_b64, "mask": mask_b64, "prompt": "a glowing blue sphere",
     "strength": 0.9, "width": 256, "height": 256, "steps": 4, "mode": "add"}, out))
t1.start()
time.sleep(1.0)
t2.start()

h = wait_for(lambda s: s["queue"] >= 1, timeout=120)
expect("queue grows with a second request", h["queue"] >= 1,
       f"busy={h['busy']} queue={h['queue']}")

t1.join(600)
t2.join(600)
expect("both requests completed", out.get("gen") == 200 and out.get("inpaint") == 200,
       f"statuses={out}")

h = wait_for(lambda s: s["busy"] is False, timeout=30)
expect("idle after completion", h["busy"] is False and h["queue"] == 0,
       f"busy={h['busy']} queue={h['queue']}")

print("\n=== FAILED REQUEST ===")
r = c.post("/v1/images/inpaint", json={
    "image": "!!!not-base64!!!", "mask": "!!!not-base64!!!",
    "prompt": "x", "width": 256, "height": 256, "mode": "add",
})
expect("invalid request rejected", r.status_code >= 400, f"status={r.status_code}")
h = wait_for(lambda s: s["busy"] is False, timeout=30)
expect("state reset after failure", h["busy"] is False and h["queue"] == 0,
       f"busy={h['busy']} queue={h['queue']}")

print("\n=== SUMMARY ===")
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    raise SystemExit(1)
print("PASS: Android health states match real request flow")
