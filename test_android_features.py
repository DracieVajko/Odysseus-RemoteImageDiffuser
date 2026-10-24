import base64
import tempfile
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8199"
c = httpx.Client(base_url=BASE, timeout=1200.0)
TD = Path(r"D:\AI\DiffusionServer\testdata")
OUT = Path(tempfile.mkdtemp(prefix="prism_android_test_"))
results = []


def check(name, method, path, expected=(200,), **kw):
    t0 = time.time()
    try:
        r = c.request(method, path, **kw)
        dt = time.time() - t0
        ok = r.status_code in expected
        body = r.text[:120].replace("\n", " ")
        results.append((name, r.status_code, ok, dt, body))
        print(f"[{'OK ' if ok else 'FAIL'}] {name:<32} {r.status_code} {dt:6.2f}s {body[:70]}")
        return r
    except Exception as e:
        results.append((name, 0, False, time.time() - t0, str(e)))
        print(f"[FAIL] {name:<32} EXC {e}")
        return None


def expect(name, cond, detail=""):
    """Record an assertion result without an HTTP call."""
    results.append((name, 0, bool(cond), 0.0, detail))
    print(f"[{'OK ' if cond else 'FAIL'}] {name:<32} {detail}")


print("=== SYSTEM ===")
check("health", "GET", "/health")
check("models", "GET", "/v1/models")
check("catalog", "GET", "/v1/models/catalog")
check("settings", "GET", "/v1/settings")

print("\n=== GENERATE ===")
r = check("generate 256 (neg, g=1)", "POST", "/v1/images/generations", json={
    "prompt": "a red cube on a wooden table", "negative_prompt": "blurry, low quality",
    "guidance": 1.0, "size": "256x256", "n": 1, "steps": 4})
if r is not None and r.status_code == 200:
    b = base64.b64decode(r.json()["data"][0]["b64_json"])
    (OUT / "android_gen_256.png").write_bytes(b)
    print(f"      bytes={len(b)}")

check("generate 384 (neg, g=2)", "POST", "/v1/images/generations", json={
    "prompt": "a blue sphere on white", "negative_prompt": "blurry, distorted",
    "guidance": 2.0, "size": "384x384", "n": 1, "steps": 4})

print("\n=== REGRESSIONS ===")
r = check("regression n=2 non-square", "POST", "/v1/images/generations", json={
    "prompt": "a red cube", "size": "256x128", "n": 2, "steps": 4})
if r is not None and r.status_code == 200:
    j = r.json()
    d = j.get("data", [])
    expect("  n=2 returns 2 images", len(d) == 2, f"count={len(d)}")
    expect("  n=2 images are distinct", len(d) == 2 and d[0]["b64_json"] != d[1]["b64_json"])
    expect("  actual size reported", j.get("size") == "256x256", f"size={j.get('size')}")
    expect("  requested size reported", j.get("requested_size") == "256x128",
           f"requested_size={j.get('requested_size')}")

print("\n=== EDITS / INPAINT / HARMONIZE ===")
img_bytes = (TD / "images" / "test_512.png").read_bytes()
img_b64 = base64.b64encode(img_bytes).decode()
mask_b64 = base64.b64encode((TD / "images" / "mask_512.png").read_bytes()).decode()

check("edits (multipart)", "POST", "/v1/images/edits", (200, 503),
      files={"image": ("test.png", img_bytes, "image/png")},
      data={"prompt": "make it blue", "size": "256x256", "quality": "low", "strength": "0.6"})

check("inpaint add", "POST", "/v1/images/inpaint", (200, 503), json={
    "image": img_b64, "mask": mask_b64, "prompt": "a glowing blue sphere",
    "strength": 0.9, "width": 256, "height": 256, "steps": 4, "mode": "add"})
check("inpaint remove", "POST", "/v1/images/inpaint", (200, 503), json={
    "image": img_b64, "mask": mask_b64, "prompt": "empty white background",
    "strength": 0.9, "width": 256, "height": 256, "steps": 4, "mode": "remove"})
check("inpaint color", "POST", "/v1/images/inpaint", (200, 503), json={
    "image": img_b64, "mask": mask_b64, "prompt": "",
    "width": 256, "height": 256, "mode": "color", "color": "#0066ff"})

check("harmonize mask", "POST", "/v1/images/harmonize", (200, 503), json={
    "image": img_b64, "mask": mask_b64, "color_match": True, "seam_fix": True, "strength": 0.6})
check("harmonize no mask", "POST", "/v1/images/harmonize", (200, 503), json={"image": img_b64})

print("\n=== AUDIO ===")
check("transcribe", "POST", "/v1/audio/transcriptions", (200,),
      files={"file": ("jfk.wav", (TD / "audio" / "jfk.wav").read_bytes(), "audio/wav")},
      data={"language": "en"})

print("\n=== UNIMPLEMENTED (expected 503) ===")
check("caption", "POST", "/v1/images/caption", (503,), json={"image": img_b64})
check("upscale", "POST", "/v1/images/upscale", (503,), json={"image": img_b64, "scale": 2})
check("ffmpeg presets", "GET", "/v1/ffmpeg/presets", (503,))

print("\n=== SUMMARY ===")
ok = sum(1 for _, _, o, _, _ in results if o)
print(f"{ok}/{len(results)} passed")
for n, code, o, dt, body in results:
    if not o:
        print(f"  FAIL {n}: {code} {body}")
