import base64
import io
import tempfile
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8200"
c = httpx.Client(base_url=BASE, timeout=900.0)
TD = Path(r"D:\AI\DiffusionServer\testdata")
OUT = Path(tempfile.mkdtemp(prefix="prism_pc_test_"))
results = []


def check(name, method, path, expected=(200,), **kw):
    t0 = time.time()
    try:
        r = c.request(method, path, **kw)
        dt = time.time() - t0
        ok = r.status_code in expected
        body = r.text[:140].replace("\n", " ")
        results.append((name, r.status_code, ok, dt, body))
        print(f"[{'OK ' if ok else 'FAIL'}] {name:<34} {r.status_code} {dt:6.2f}s {body[:80]}")
        return r
    except Exception as e:
        results.append((name, 0, False, time.time() - t0, str(e)))
        print(f"[FAIL] {name:<34} EXC {e}")
        return None


def expect(name, cond, detail=""):
    """Record an assertion result without an HTTP call."""
    results.append((name, 0, bool(cond), 0.0, detail))
    print(f"[{'OK ' if cond else 'FAIL'}] {name:<34} {detail}")


print("=== SYSTEM ===")
check("health", "GET", "/health")
check("models", "GET", "/v1/models")
check("catalog", "GET", "/v1/models/catalog")
check("settings", "GET", "/v1/settings")
r = check("ffmpeg presets", "GET", "/v1/ffmpeg/presets")
if r is not None:
    p = r.json()
    print(f"      video={len(p.get('video',{}))} audio={len(p.get('audio',{}))} image={len(p.get('image',{}))}")

print("\n=== TEXT TO IMAGE ===")
r = check("generate 256 (neg, g=1)", "POST", "/v1/images/generations", json={
    "prompt": "a red cube on a wooden table", "negative_prompt": "blurry, low quality",
    "guidance": 1.0, "size": "256x256", "quality": "low", "n": 1})
if r is not None and r.status_code == 200:
    b = base64.b64decode(r.json()["data"][0]["b64_json"])
    (OUT / "pc_test_gen_256.png").write_bytes(b)
    print(f"      bytes={len(b)}")

check("generate 384 (neg, g=2)", "POST", "/v1/images/generations", json={
    "prompt": "a blue sphere on white", "negative_prompt": "blurry, distorted",
    "guidance": 2.0, "size": "384x384", "quality": "medium", "n": 1})

check("generate custom 320x256", "POST", "/v1/images/generations", json={
    "prompt": "a green tree", "size": "320x256", "quality": "low", "n": 1})

print("\n=== REGRESSIONS ===")
r = check("regression n=2", "POST", "/v1/images/generations", json={
    "prompt": "a red cube", "size": "256x256", "quality": "low", "n": 2})
if r is not None and r.status_code == 200:
    d = r.json()["data"]
    expect("  n=2 returns 2 images", len(d) == 2, f"count={len(d)}")
    expect("  n=2 images are distinct", len(d) == 2 and d[0]["b64_json"] != d[1]["b64_json"])

r = check("regression quality=high", "POST", "/v1/images/generations", json={
    "prompt": "a red cube", "size": "256x256", "quality": "high", "n": 1})
if r is not None and r.status_code == 200:
    j = r.json()
    expect("  high maps to 8 steps", j.get("steps") == 8, f"steps={j.get('steps')}")
    expect("  quality echoed", j.get("quality") == "high", f"quality={j.get('quality')}")

r = check("regression size metadata", "POST", "/v1/images/generations", json={
    "prompt": "a green tree", "size": "320x256", "quality": "low", "n": 1})
if r is not None and r.status_code == 200:
    j = r.json()
    expect("  actual size reported", j.get("size") == "320x320", f"size={j.get('size')}")
    expect("  requested size reported", j.get("requested_size") == "320x256",
           f"requested_size={j.get('requested_size')}")

print("\n=== IMAGE TO IMAGE ===")
img_b64 = base64.b64encode((TD / "images" / "test_512.png").read_bytes()).decode()
r = check("img2img (edits)", "POST", "/v1/images/edits",
          files={"image": ("test.png", (TD / "images" / "test_512.png").read_bytes(), "image/png")},
          data={"prompt": "make it blue", "negative_prompt": "blurry", "guidance": "1.0",
                "size": "256x256", "quality": "low", "strength": "0.6"})
if r is not None and r.status_code == 200:
    d = r.json()["data"][0]["b64_json"]
    (OUT / "pc_test_img2img.png").write_bytes(base64.b64decode(d))
    print(f"      bytes={len(d)}")

print("\n=== INPAINT ===")
mask_b64 = base64.b64encode((TD / "images" / "mask_512.png").read_bytes()).decode()
r = check("inpaint add", "POST", "/v1/images/inpaint", json={
    "image": img_b64, "mask": mask_b64, "prompt": "a glowing blue sphere",
    "negative_prompt": "blurry", "guidance": 1.0, "strength": 0.9,
    "width": 256, "height": 256, "steps": 4, "mode": "add"})
if r is not None and r.status_code == 200:
    (OUT / "pc_test_inpaint_add.png").write_bytes(base64.b64decode(r.json()["image"]))
    print(f"      engine={r.json().get('engine')}")

r = check("inpaint remove", "POST", "/v1/images/inpaint", json={
    "image": img_b64, "mask": mask_b64, "prompt": "empty white background, clean surface",
    "strength": 0.9, "width": 256, "height": 256, "steps": 4, "mode": "remove"})
if r is not None and r.status_code == 200:
    (OUT / "pc_test_inpaint_remove.png").write_bytes(base64.b64decode(r.json()["image"]))

r = check("inpaint color", "POST", "/v1/images/inpaint", json={
    "image": img_b64, "mask": mask_b64, "prompt": "",
    "width": 512, "height": 512, "mode": "color", "color": "#0066ff"})
if r is not None and r.status_code == 200:
    (OUT / "pc_test_inpaint_color.png").write_bytes(base64.b64decode(r.json()["image"]))
    print(f"      engine={r.json().get('engine')}")

print("\n=== UPSCALE / VISION ===")
r = check("upscale 2x", "POST", "/v1/images/upscale", json={"image": img_b64, "scale": 2})
if r is not None and r.status_code == 200:
    expect("  realesrgan engine", str(r.json().get("engine", "")).startswith("realesrgan"),
           f"engine={r.json().get('engine')}")

r = check("caption", "POST", "/v1/images/caption", json={"image": img_b64, "max_tokens": 32})
if r is not None and r.status_code == 200:
    expect("  caption has text", bool(r.json().get("text")), r.json().get("text", "")[:50])

r = check("chat vision", "POST", "/v1/chat/completions", json={
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + img_b64}},
        {"type": "text", "text": "What is in the image?"},
    ]}], "max_tokens": 32})
if r is not None and r.status_code == 200:
    content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    expect("  chat has content", bool(content), content[:50])

print("\n=== FFMPEG ===")
check("convert-image png->jpg", "POST", "/v1/ffmpeg/convert-image",
      files={"file": ("test.png", (TD / "images" / "test_512.png").read_bytes(), "image/png")},
      data={"preset": "jpg-95"})
check("convert-audio wav->mp3", "POST", "/v1/ffmpeg/convert-audio",
      files={"file": ("tone.wav", (TD / "audio" / "tone_3s.wav").read_bytes(), "audio/wav")},
      data={"preset": "mp3-192"})
check("convert-video mp4->h264-fast", "POST", "/v1/ffmpeg/convert-video",
      files={"file": ("test.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")},
      data={"preset": "mp4-h264-fast"})
check("extract-audio", "POST", "/v1/ffmpeg/extract-audio",
      files={"file": ("test.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")},
      data={"format": "wav"})
check("trim video", "POST", "/v1/ffmpeg/trim",
      files={"file": ("test.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")},
      data={"start": "1", "end": "3"})
check("media info", "POST", "/v1/ffmpeg/info",
      files={"file": ("test.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")})
check("compress video", "POST", "/v1/ffmpeg/compress",
      files={"file": ("test.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")},
      data={"target_size_mb": "2"})

print("\n=== AUDIO ===")
check("transcribe wav", "POST", "/v1/audio/transcriptions",
      files={"file": ("jfk.wav", (TD / "audio" / "jfk.wav").read_bytes(), "audio/wav")},
      data={"language": "en"})
r = check("translate wav", "POST", "/v1/audio/translations",
          files={"file": ("jfk.wav", (TD / "audio" / "jfk.wav").read_bytes(), "audio/wav")})
if r is not None and r.status_code == 200:
    expect("  translation has text", bool(r.json().get("text")), r.json().get("text", "")[:40])

print("\n=== SUMMARY ===")
ok = sum(1 for _, _, o, _, _ in results if o)
print(f"{ok}/{len(results)} passed")
for n, code, o, dt, body in results:
    if not o:
        print(f"  FAIL {n}: {code} {body}")
