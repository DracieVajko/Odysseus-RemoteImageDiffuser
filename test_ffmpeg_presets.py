import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8200"
c = httpx.Client(base_url=BASE, timeout=600.0)
TD = Path(r"D:\AI\DiffusionServer\testdata")
VIDEO = (TD / "video" / "test_5s.mp4").read_bytes()
AUDIO = (TD / "audio" / "tone_3s.wav").read_bytes()
IMAGE = (TD / "images" / "test_512.png").read_bytes()

results = []


def check(name, path, files, data):
    t0 = time.time()
    try:
        r = c.post(path, files=files, data=data)
        dt = time.time() - t0
        ok = r.status_code == 200
        body = r.json() if ok else r.text[:120]
        results.append((name, r.status_code, ok, dt, body))
        size = len(body.get("file", "")) if isinstance(body, dict) else 0
        print(f"[{'OK ' if ok else 'FAIL'}] {name:<28} {r.status_code} {dt:6.2f}s  b64={size}")
        return body
    except Exception as e:
        results.append((name, 0, False, time.time() - t0, str(e)))
        print(f"[FAIL] {name:<28} EXC {e}")
        return None


print("=== FFMPEG VIDEO PRESETS ===")
for preset in ["mp4-h264", "mp4-h264-hq", "mp4-h264-fast",
               "mp4-h265", "mp4-h265-hq", "mp4-h265-fast",
               "mp4-small", "mp4-discord", "webm", "gif"]:
    check(f"video {preset}", "/v1/ffmpeg/convert-video",
          files={"file": ("test.mp4", VIDEO, "video/mp4")},
          data={"preset": preset})

print("\n=== FFMPEG AUDIO PRESETS ===")
for preset in ["mp3-128", "mp3-192", "mp3-320", "aac-128", "aac-192",
               "wav-16k", "opus", "flac"]:
    check(f"audio {preset}", "/v1/ffmpeg/convert-audio",
          files={"file": ("tone.wav", AUDIO, "audio/wav")},
          data={"preset": preset})

print("\n=== FFMPEG IMAGE PRESETS ===")
for preset in ["jpg-95", "jpg-85", "png", "webp", "avif"]:
    check(f"image {preset}", "/v1/ffmpeg/convert-image",
          files={"file": ("test.png", IMAGE, "image/png")},
          data={"preset": preset})

print("\n=== SUMMARY ===")
ok = sum(1 for _, _, o, _, _ in results if o)
print(f"{ok}/{len(results)} passed")
for n, code, o, dt, body in results:
    if not o:
        print(f"  FAIL {n}: {code} {body}")
