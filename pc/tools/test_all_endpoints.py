import base64
import tempfile
import json
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8199"
client = httpx.Client(base_url=BASE, timeout=900.0)

results = []


def check(name, method, path, expected=None, **kwargs):
    t0 = time.time()
    try:
        r = client.request(method, path, **kwargs)
        dt = time.time() - t0
        ok = expected is None or r.status_code in expected
        results.append((name, r.status_code, ok, dt, r.text[:160]))
        print(f"[{'OK ' if ok else 'FAIL'}] {name:<32} {r.status_code} {dt:6.2f}s  {r.text[:100]}")
        return r
    except Exception as e:
        results.append((name, 0, False, time.time() - t0, str(e)))
        print(f"[FAIL] {name:<32} EXC {e}")
        return None


print("=== HEALTH / MODELS ===")
check("health", "GET", "/health", (200,))
check("models", "GET", "/v1/models", (200,))
check("catalog", "GET", "/v1/models/catalog", (200,))
check("custom", "GET", "/v1/models/custom", (200,))
check("settings", "GET", "/v1/settings", (200,))
check("progress (installed)", "GET", "/v1/models/progress/whisper-large-v3-turbo-q5_0", (200,))
check("progress (unknown)", "GET", "/v1/models/progress/nope", (200, 404))

print("\n=== AUDIO ===")
wav = (ROOT / "android" / "app" / "src" / "main" / "cpp" / "third_party" / "whisper.cpp" / "samples" / "jfk.wav").read_bytes()
check("transcribe (en)", "POST", "/v1/audio/transcriptions", (200,),
      files={"file": ("jfk.wav", wav, "audio/wav")}, data={"language": "en"})
check("translate", "POST", "/v1/audio/translations", (200,),
      files={"file": ("jfk.wav", wav, "audio/wav")})

print("\n=== IMAGE GENERATION ===")
r = check("generate 256x256", "POST", "/v1/images/generations", (200,),
          json={"prompt": "a red cube on a wooden table", "size": "256x256", "n": 1, "steps": 4})
gen_png = None
if r is not None and r.status_code == 200:
    b64 = r.json().get("data", [{}])[0].get("b64_json", "")
    gen_png = base64.b64decode(b64)
    (Path(tempfile.mkdtemp(prefix="prism_e2e_")) / "test_generate.png").write_bytes(gen_png)
    print(f"      image bytes={len(gen_png)}")

check("generate 512x512", "POST", "/v1/images/generations", (200,),
      json={"prompt": "a blue sphere", "size": "512x512", "n": 1, "steps": 4})

print("\n=== IMAGE EDITS / INPAINT / HARMONIZE ===")
if gen_png:
    check("edits (multipart)", "POST", "/v1/images/edits", (200, 502, 503),
          files={"image": ("in.png", gen_png, "image/png")},
          data={"prompt": "make it blue", "size": "256x256"})
    b64 = base64.b64encode(gen_png).decode()
    check("inpaint", "POST", "/v1/images/inpaint", (200, 502, 503),
          json={"image": b64, "mask": b64, "prompt": "remove", "width": 256, "height": 256})
    check("harmonize", "POST", "/v1/images/harmonize", (200, 502, 503),
          json={"image": b64, "prompt": "match colors", "max_side": 512})
    check("upscale 2x", "POST", "/v1/images/upscale", (200, 502, 503),
          json={"image": b64, "scale": 2})
    check("caption", "POST", "/v1/images/caption", (200, 502, 503),
          json={"image": b64})
else:
    print("[SKIP] no generated image")

print("\n=== CHAT / FFMPEG / WEBUI ===")
check("chat completions", "POST", "/v1/chat/completions", (200, 502, 503),
      json={"model": "smolvlm", "messages": [{"role": "user", "content": "hi"}]})
check("ffmpeg presets", "GET", "/v1/ffmpeg/presets", (200, 502, 503))
check("ffmpeg info (multipart)", "POST", "/v1/ffmpeg/info", (200, 502, 503),
      files={"file": ("jfk.wav", wav, "audio/wav")})
check("webui removed", "GET", "/webui", (404,))

print("\n=== MODEL MANAGEMENT ===")
check("install (already installed)", "POST", "/v1/models/install", (200, 409),
      json={"id": "whisper-large-v3-turbo-q5_0"})
check("cancel (idle)", "POST", "/v1/models/cancel", (200,),
      json={"id": "whisper-large-v3-turbo-q5_0"})

print("\n=== VALIDATION ===")
check("generate bad size", "POST", "/v1/images/generations", (400,),
      json={"prompt": "x", "size": "abc"})
check("generate n=5", "POST", "/v1/images/generations", (400,),
      json={"prompt": "x", "size": "256x256", "n": 5})
check("transcribe empty", "POST", "/v1/audio/transcriptions", (400,),
      files={"file": ("e.wav", b"", "audio/wav")})

print("\n=== SUMMARY ===")
ok = sum(1 for _, _, o, _, _ in results if o)
print(f"{ok}/{len(results)} checks passed")
for name, code, o, dt, body in results:
    if not o:
        print(f"  FAIL {name}: {code} {body}")
