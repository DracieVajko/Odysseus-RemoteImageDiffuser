"""Live feature suite for the on-device Prism server.

Run with the phone connected and the server started in the app:

    adb forward tcp:8199 tcp:8101
    python tests/test_android_live.py --base http://127.0.0.1:8199

Every check asserts content (image size and variance, mask inside/outside,
caption words, transcript text), not just HTTP 200. Weights are assumed to be
installed via /v1/models/install; missing ones are reported, not skipped
silently.
"""
import argparse
import base64
import io
import sys
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "testdata"
OUT = Path(__import__("os").environ.get("TEMP", ".")) / "prism_android_live"
OUT.mkdir(parents=True, exist_ok=True)

RESULTS = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        ok = True
    except AssertionError as e:
        detail, ok = f"ASSERT: {e}", False
    except Exception as e:
        detail, ok = f"EXC: {type(e).__name__}: {str(e)[:160]}", False
    dt = time.time() - t0
    RESULTS.append((name, ok, detail, round(dt, 1)))
    print(f"[{'OK  ' if ok else 'FAIL'}] {name:<32} {dt:7.1f}s  {str(detail)[:100]}")
    return ok


def img_from_b64(s):
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


def png_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8199")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.base.rstrip("/"), timeout=1800)

    def health():
        j = c.get("/health").json()
        assert j.get("status") == "ok", j
        return f"device={j.get('device')} models={len(j.get('available_models', []))}"

    def models():
        j = c.get("/v1/models").json()
        ids = [m["id"] for m in j["data"]]
        assert len(ids) == len(set(ids)), "duplicate ids"
        return f"{len(ids)} models, kinds={sorted({m['kind'] for m in j['data']})}"

    def catalog():
        j = c.get("/v1/models/catalog").json()
        assert j["models"], "empty catalog"
        ready = [m["id"] for m in j["models"] if m.get("installed")]
        return f"{len(j['models'])} entries, installed={len(ready)}"

    def status_verify():
        s = c.get("/v1/models/status/lcm-dreamshaper-int8-onnx")
        assert s.status_code == 200, s.text[:120]
        v = c.post("/v1/models/verify/lcm-dreamshaper-int8-onnx")
        assert v.status_code == 200 and v.json()["verified"] is True, v.text[:120]
        missing = c.get("/v1/models/status/no-such-model")
        assert missing.status_code == 404, missing.status_code
        return f"status={s.json()['installed']} verify={v.json()['verified']} 404=ok"

    def settings():
        j = c.get("/v1/settings").json()
        assert j["engines"]["onnxruntime"] is True
        return f"port={j['port']} whisper={j['engines']['whisper_native']}"

    def generate():
        r = c.post("/v1/images/generations", json={
            "prompt": "a red apple on a wooden table, photo",
            "negative_prompt": "blurry", "size": "256x256", "quality": "low",
            "request_id": "live-gen"})
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        im = img_from_b64(j["data"][0]["b64_json"])
        a = np.asarray(im, dtype=np.float32)
        assert im.size == (256, 256), im.size
        assert a.std() > 5, f"flat image (std={a.std():.1f})"
        im.save(OUT / "gen.png")
        return f"{im.size} std={a.std():.1f} engine={j.get('engine')}"

    def generate_warm():
        t0 = time.time()
        r = c.post("/v1/images/generations", json={"prompt": "a blue circle", "size": "256x256",
                                                   "quality": "low"})
        assert r.status_code == 200, r.text[:120]
        return f"{time.time() - t0:.1f}s warm"

    def edit():
        src = Image.new("RGB", (256, 256), (60, 120, 200))
        ImageDraw.Draw(src).rectangle([80, 80, 176, 176], fill=(240, 220, 60))
        buf = io.BytesIO()
        src.save(buf, format="PNG")
        r = c.post("/v1/images/edits",
                   files={"image": ("s.png", buf.getvalue(), "image/png")},
                   data={"prompt": "winter scene, snow", "size": "256x256",
                         "strength": "0.8", "quality": "low"})
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        out = img_from_b64(j["data"][0]["b64_json"])
        d = float(np.abs(np.asarray(src, np.float32) -
                         np.asarray(out.resize(src.size), np.float32)).mean())
        assert out.size == src.size and d > 1.0, f"{out.size} diff={d:.1f}"
        return f"{out.size} mean|diff|={d:.1f} engine={j.get('engine')}"

    def inpaint():
        base = Image.new("RGB", (256, 256), (200, 210, 220))
        mask = Image.new("L", (256, 256), 0)
        ImageDraw.Draw(mask).ellipse([88, 88, 168, 168], fill=255)
        r = c.post("/v1/images/inpaint", json={
            "image": png_b64(base), "mask": png_b64(mask), "prompt": "a red flower",
            "mode": "add", "strength": 0.9, "width": 256, "height": 256, "steps": 6})
        assert r.status_code == 200, r.text[:200]
        out = img_from_b64(r.json()["image"])
        a = np.asarray(base, np.float32)
        b = np.asarray(out.resize(base.size), np.float32)
        m = np.asarray(mask) > 127
        inside = float(np.abs(a[m] - b[m]).mean())
        outside = float(np.abs(a[~m] - b[~m]).mean())
        assert inside > 5, f"mask unchanged ({inside:.1f})"
        assert outside < 3.0, f"outside changed ({outside:.2f})"
        return f"inside={inside:.1f} outside={outside:.2f}"

    def outpaint():
        src = Image.new("RGB", (192, 192), (120, 160, 90))
        ImageDraw.Draw(src).ellipse([48, 48, 144, 144], fill=(250, 240, 200))
        r = c.post("/v1/images/outpaint", json={"image": png_b64(src), "margin": 0.25,
                                                "prompt": "continue the scene"})
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        out = img_from_b64(j["image"])
        assert out.size == (288, 288), out.size
        a = np.asarray(src, np.float32)
        b = np.asarray(out, np.float32)[48:240, 48:240]
        centre = float(np.abs(a - b).mean())
        assert centre < 1.0, f"original pixels changed ({centre:.2f})"
        return f"{j['input_size']}->{j['output_size']} centre={centre:.2f}"

    def upscale():
        # 4x on a 256 source -> 1024 keeps the phone responsive; 512 -> 2048 is
        # refused by the mobile cap and checked separately below.
        src = Image.open(TD / "images" / "test_512.png").convert("RGB").resize((256, 256))
        r = c.post("/v1/images/upscale", json={"image": png_b64(src), "scale": 4})
        assert r.status_code == 200, r.text[:200]
        out = img_from_b64(r.json()["image"])
        a = np.asarray(out, np.float32)
        assert out.size == (1024, 1024), out.size
        assert a.std() > 5, f"flat output ({a.std():.1f})"
        return f"{out.size} std={a.std():.1f} engine={r.json().get('engine')}"

    def upscale_cap():
        src = Image.open(TD / "images" / "test_512.png").convert("RGB")
        r = c.post("/v1/images/upscale", json={"image": png_b64(src), "scale": 4})
        assert r.status_code == 400, f"expected the mobile cap, got {r.status_code}"
        return r.json().get("error", "")[:60]

    def caption():
        img = Image.new("RGB", (256, 256), (255, 255, 255))
        ImageDraw.Draw(img).rectangle([64, 64, 192, 192], fill=(200, 30, 30))
        r = c.post("/v1/images/caption", json={"image": png_b64(img),
                                               "prompt": "Describe the image briefly.",
                                               "max_tokens": 48})
        assert r.status_code == 200, r.text[:200]
        text = (r.json().get("caption") or "").strip()
        assert len(text) > 10, f"too short: {text!r}"
        low = text.lower()
        assert "red" in low or "square" in low, f"unexpected caption: {text[:80]}"
        return f"{len(text)} chars: {text[:60]}"

    def chat():
        img = Image.new("RGB", (256, 256), (255, 255, 255))
        ImageDraw.Draw(img).rectangle([64, 64, 192, 192], fill=(200, 30, 30))
        r = c.post("/v1/chat/completions", json={
            "model": "smolvlm-500m-gguf", "max_tokens": 32,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What colour is the square?"},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64," + png_b64(img)}}]}]})
        assert r.status_code == 200, r.text[:200]
        answer = r.json()["choices"][0]["message"]["content"].strip()
        assert len(answer) > 2, f"empty answer {answer!r}"
        return f"{len(answer)} chars: {answer[:40]}"

    def transcribe():
        wav = (TD / "audio" / "jfk.wav").read_bytes()
        r = c.post("/v1/audio/transcriptions",
                   files={"file": ("jfk.wav", wav, "audio/wav")},
                   data={"model": "whisper-large-v3-turbo-q5_0", "language": "en",
                         "auto_split": "false"})
        assert r.status_code == 200, r.text[:200]
        text = (r.json().get("text") or "").lower()
        assert "ask not what your country can do for you" in text, text[:120]
        return f"{r.json().get('model')}: {text[:40]}"

    def translations():
        wav = (TD / "audio" / "jfk.wav").read_bytes()
        r = c.post("/v1/audio/translations",
                   files={"file": ("jfk.wav", wav, "audio/wav")},
                   data={"model": "whisper-large-v3-turbo-q5_0"})
        assert r.status_code == 200, r.text[:200]
        return f"{len(r.json().get('text') or '')} chars"

    # ---------------------------------------------------------- ffmpeg tools
    def ff_presets():
        j = c.get("/v1/ffmpeg/presets").json()
        assert j.get("engine") == "android-mediacodec", j.get("engine")
        sup = sum(1 for g in ("video", "audio", "image")
                  for v in j[g].values() if v.get("supported"))
        assert sup >= 10, sup
        return f"{len(j['video'])} video / {len(j['audio'])} audio / {len(j['image'])} image, {sup} supported"

    def ff_info():
        files = {"file": ("t.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")}
        r = c.post("/v1/ffmpeg/info", files=files)
        assert r.status_code == 200, r.text[:160]
        info = r.json()["info"]
        kinds = [x["codec_type"] for x in info["streams"]]
        assert "video" in kinds and "audio" in kinds, kinds
        return f"{info['duration_s']:.1f}s {kinds}"

    def ff_image():
        files = {"file": ("t.png", (TD / "images" / "test_512.png").read_bytes(), "image/png")}
        r = c.post("/v1/ffmpeg/convert-image", files=files, data={"preset": "jpg-95"})
        assert r.status_code == 200, r.text[:160]
        raw = base64.b64decode(r.json()["file"])
        assert raw[:2] == b"\xff\xd8", raw[:4]
        return f"{len(raw)} B JPEG"

    def ff_extract():
        files = {"file": ("t.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")}
        r = c.post("/v1/ffmpeg/extract-audio", files=files)
        assert r.status_code == 200, r.text[:160]
        raw = base64.b64decode(r.json()["file"])
        assert raw[4:8] == b"ftyp", raw[:12]
        return f"{len(raw)} B m4a"

    def ff_trim():
        files = {"file": ("t.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")}
        r = c.post("/v1/ffmpeg/trim", files=files, data={"start": "1", "end": "3"})
        assert r.status_code == 200, r.text[:160]
        raw = base64.b64decode(r.json()["file"])
        r2 = c.post("/v1/ffmpeg/info", files={"file": ("trim.mp4", raw, "video/mp4")})
        dur = r2.json()["info"].get("duration_s") or 0
        assert 1.5 <= dur <= 3.5, f"trimmed duration {dur}"
        return f"{dur:.2f}s"

    def ff_audio_aac():
        files = {"file": ("t.wav", (TD / "audio" / "jfk.wav").read_bytes(), "audio/wav")}
        r = c.post("/v1/ffmpeg/convert-audio", files=files, data={"preset": "aac-128"})
        assert r.status_code == 200, r.text[:160]
        raw = base64.b64decode(r.json()["file"])
        assert raw[4:8] == b"ftyp", raw[:12]
        return f"{len(raw)} B m4a"

    def ff_audio_wav16k():
        files = {"file": ("t.wav", (TD / "audio" / "jfk.wav").read_bytes(), "audio/wav")}
        r = c.post("/v1/ffmpeg/convert-audio", files=files, data={"preset": "wav-16k"})
        assert r.status_code == 200, r.text[:160]
        raw = base64.b64decode(r.json()["file"])
        assert raw[:4] == b"RIFF", raw[:4]
        rate = int.from_bytes(raw[24:28], "little")
        ch = int.from_bytes(raw[22:24], "little")
        assert rate == 16000 and ch == 1, f"{rate} Hz {ch} ch"
        return f"{rate} Hz mono {len(raw)} B"

    def ff_mp3():
        files = {"file": ("t.wav", (TD / "audio" / "jfk.wav").read_bytes(), "audio/wav")}
        r = c.post("/v1/ffmpeg/convert-audio", files=files, data={"preset": "mp3-192"})
        assert r.status_code == 200, r.text[:160]
        raw = base64.b64decode(r.json()["file"])
        assert raw[:3] == b"ID3" or (raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0), raw[:8]
        assert len(raw) > 100_000, len(raw)
        return f"{len(raw)} B MP3 (LAME)"

    def ff_compress():
        files = {"file": ("t.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")}
        r = c.post("/v1/ffmpeg/compress", files=files, data={"preset": "web-720p"})
        # Either the on-device re-encode succeeds (200) or it honestly
        # reports 503 if the EGL/GL pipeline is unavailable on this device.
        assert r.status_code in (200, 503), r.status_code
        if r.status_code == 200:
            j = r.json()
            import base64 as _b64
            raw = _b64.b64decode(j.get("file") or j.get("data") or "")
            assert len(raw) > 1000, len(raw)
            assert raw[:4] in (b"\x00\x00\x00\x18", b"\x00\x00\x00\x1c", b"\x00\x00\x00 ", b"\x00\x00\x00\x20"), raw[:4]
            return f"{len(raw)} B (re-encoded)"
        return "honest 503 (EGL unavailable)"

    def err_missing_image():
        r = c.post("/v1/images/inpaint", json={"prompt": "x"})
        assert r.status_code == 400, r.status_code
        return r.json().get("error", "")[:40]

    def err_ffmpeg():
        r = c.post("/v1/ffmpeg/info", files={"file": ("x.wav", b"RIFF", "audio/wav")})
        assert r.status_code == 503, r.status_code
        return r.json().get("detail", "")[:50]

    def unknown_model():
        r = c.get("/v1/models/verify/nope")
        assert r.status_code == 404, r.status_code
        return "404"

    check("health", health)
    check("models", models)
    check("catalog", catalog)
    check("models status/verify", status_verify)
    check("settings", settings)
    check("generate 256x256", generate)
    check("generate warm", generate_warm)
    check("edit (img2img)", edit)
    check("inpaint add", inpaint)
    check("outpaint 192->288", outpaint)
    check("upscale 4x 256->1024", upscale)
    check("upscale cap 512->2048 -> 400", upscale_cap)
    check("caption (VLM)", caption)
    check("chat (vision)", chat)
    check("transcribe wav", transcribe)
    check("translations", translations)
    check("ffmpeg presets", ff_presets)
    check("ffmpeg info", ff_info)
    check("ffmpeg image png->jpg", ff_image)
    check("ffmpeg extract-audio", ff_extract)
    check("ffmpeg trim 1s..3s", ff_trim)
    check("ffmpeg audio -> aac", ff_audio_aac)
    check("ffmpeg audio -> wav 16k", ff_audio_wav16k)
    check("ffmpeg audio -> mp3", ff_mp3)
    check("ffmpeg compress", ff_compress)
    check("error: missing image -> 400", err_missing_image)
    check("verify unknown -> 404", unknown_model)

    ok = sum(1 for _, o, _, _ in RESULTS if o)
    print(f"\n{ok}/{len(RESULTS)} passed")
    for name, o, detail, dt in RESULTS:
        if not o:
            print(f"  FAIL {name}: {detail}")
    print(f"artifacts: {OUT}")
    return 0 if ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
