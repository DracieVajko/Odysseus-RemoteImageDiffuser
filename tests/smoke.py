"""Minimal round-trip check: 1 generation with quality=low (4 steps)."""
import base64, json, os, sys, time, urllib.request
BASE = "http://127.0.0.1:8100"
model = sys.argv[1] if len(sys.argv) > 1 else "DreamShaper_8"
outdir = r"D:\AI\DiffusionServer\tests\output"
os.makedirs(outdir, exist_ok=True)
with urllib.request.urlopen(BASE + "/health", timeout=15) as r:
    print("health:", r.read().decode(), flush=True)
body = json.dumps({"model": model, "prompt": "a small red cabin in the woods, daytime", "size": "512x512", "n": 1, "quality": "low"}).encode()
req = urllib.request.Request(BASE + "/v1/images/generations", data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=300) as r:
    res = json.loads(r.read().decode())
dt = time.time() - t0
b64 = res["data"][0]["b64_json"]
p = os.path.join(outdir, model + "-smoke.png")
with open(p, "wb") as f:
    f.write(base64.b64decode(b64))
print(f"OK in {dt:.1f}s, saved {p} ({os.path.getsize(p)} bytes)", flush=True)
