"""Full PC server test: every product endpoint with real files and real assertions.

Run against a live server:
    python tools/test_pc_all.py --base http://127.0.0.1:8102

Checks content, not only status codes: image dimensions, pixel changes inside
and outside masks, upscale factors, transcript words, ffprobe metadata.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TD = ROOT / "testdata"
OUT = Path(os.environ.get("TEMP", ".")) / "prism_full_test"
OUT.mkdir(parents=True, exist_ok=True)

results = []
BASE = "http://127.0.0.1:8102"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def img_from(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def diff(a: Image.Image, b: Image.Image) -> float:
    import numpy as np
    x = np.asarray(a, dtype=np.float32)
    y = np.asarray(b, dtype=np.float32)
    return float(abs(x - y).mean())


ONLY = ""


def check(name, fn):
    if ONLY and ONLY not in name.lower():
        return True
    t0 = time.time()
    try:
        detail = fn()
        ok = True
    except AssertionError as e:
        detail, ok = f"ASSERT: {e}", False
    except Exception as e:
        detail, ok = f"EXC: {type(e).__name__}: {str(e)[:180]}", False
    dt = time.time() - t0
    results.append({"name": name, "ok": ok, "detail": str(detail)[:300], "sec": round(dt, 2)})
    print(f"[{'OK  ' if ok else 'FAIL'}] {name:<38} {dt:7.2f}s  {str(detail)[:110]}")
    return ok


def ffprobe(path: Path) -> dict:
    exe = shutil_which("ffprobe")
    if not exe:
        return {}
    out = subprocess.run([exe, "-v", "error", "-print_format", "json",
                          "-show_format", "-show_streams", str(path)],
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout or "{}")
    except Exception:
        return {}


def shutil_which(name: str):
    for p in os.environ.get("PATH", "").split(os.pathsep):
        for ext in ("", ".exe"):
            cand = Path(p) / (name + ext)
            if cand.is_file():
                return str(cand)
    return None


def main() -> int:
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.environ.get("PRISM_BASE", BASE))
    ap.add_argument("--skip-files", action="store_true", help="skip ffmpeg/audio work")
    ap.add_argument("--only", default="", help="run only checks whose name contains this")
    args = ap.parse_args()
    global ONLY
    ONLY = args.only.strip().lower()
    BASE = args.base.rstrip("/")
    c = httpx.Client(base_url=BASE, timeout=900.0)
    print(f"target: {BASE}\n")

    # ---------------- system ----------------
    def health():
        r = c.get("/health")
        assert r.status_code == 200, r.text[:120]
        j = r.json()
        assert j.get("status") == "ok"
        return f"device={j.get('device')} provider={j.get('onnx_provider')} busy={j.get('busy')}"

    def models():
        r = c.get("/v1/models")
        j = r.json()
        ids = [m["id"] for m in j["data"]]
        assert len(ids) == len(set(ids)), "duplicate ids"
        kinds = sorted({m["kind"] for m in j["data"]})
        return f"{len(ids)} models, kinds={kinds}"

    def catalog():
        j = c.get("/v1/models/catalog").json()
        assert j["models"], "empty catalog"
        states = {m["install_state"] for m in j["models"]}
        return f"{len(j['models'])} entries, states={sorted(states)}"

    def settings():
        r = c.get("/v1/settings")
        assert r.status_code == 200, r.text[:120]
        return f"{len(r.json())} keys"

    def webui():
        r = c.get("/webui")
        assert r.status_code == 200
        assert "<title>Prism</title>" in r.text
        for tab in ("Generate", "Transcribe", "Upscale", "Caption", "Inpaint",
                    "Edit", "Tools", "Models", "Settings"):
            assert tab in r.text, f"tab {tab} missing"
        return f"{len(r.content)} bytes, 9 tabs"

    check("health", health)
    check("models list (no duplicates)", models)
    check("models catalog", catalog)
    check("settings", settings)
    check("webui html", webui)

    # ---------------- generation ----------------
    def generate():
        body = {"model": "", "prompt": "a red apple on a wooden table, photo",
                "negative_prompt": "blurry, watermark", "size": "512x512",
                "steps": 4, "guidance": 1.0, "request_id": "t-gen-1"}
        r = c.post("/v1/images/generations", json=body)
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        assert j["data"] and j["data"][0].get("b64_json"), "no data[0].b64_json"
        im = img_from(j["data"][0]["b64_json"])
        assert im.size == (512, 512), f"size {im.size}"
        assert j.get("size") == "512x512", f"reported size {j.get('size')}"
        import numpy as np
        arr = np.asarray(im)
        assert arr.std() > 5, f"image looks flat (std={arr.std():.2f})"
        (OUT / "gen_512.png").write_bytes(base64.b64decode(j["data"][0]["b64_json"]))
        return f"{im.size} std={arr.std():.1f} requested={j.get('requested_size')}"

    def generate_custom():
        body = {"prompt": "a blue circle on white background", "size": "320x256",
                "steps": 4, "request_id": "t-gen-2"}
        r = c.post("/v1/images/generations", json=body)
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        im = img_from(j["data"][0]["b64_json"])
        # the reported size must match the produced pixels, always
        assert j.get("size") == f"{im.width}x{im.height}", \
            f"reported {j.get('size')} != produced {im.size}"
        return f"produced {im.size} reported {j.get('size')} requested {j.get('requested_size')}"

    def img2img():
        src = Image.open(TD / "images" / "test_512.png").convert("RGB")
        files = {"image": ("src.png", (TD / "images" / "test_512.png").read_bytes(), "image/png")}
        data = {"prompt": "the same scene in winter, snow at night",
                "size": "512x512", "strength": "0.8", "quality": "low",
                "request_id": "t-img"}
        r = c.post("/v1/images/edits", files=files, data=data)
        assert r.status_code == 200, r.text[:300]
        j = r.json()
        payload = j.get("data", [{}])[0].get("b64_json") or j.get("image")
        assert payload, f"no image in response: {list(j)}"
        out = img_from(payload)
        assert out.size == src.size, f"size {out.size} != {src.size}"
        d = diff(src, out)
        assert d > 1.0, f"output identical to input (diff={d:.3f})"
        (OUT / "img2img.png").write_bytes(base64.b64decode(payload))
        return f"{out.size} mean|diff|={d:.1f} engine={j.get('engine')}"

    def inpaint_add():
        src = Image.open(TD / "images" / "test_512.png").convert("RGB")
        mask = Image.open(TD / "images" / "mask_512.png").convert("L")
        body = {"image": b64(TD / "images" / "test_512.png"),
                "mask": b64(TD / "images" / "mask_512.png"),
                "prompt": "a red flower", "mode": "add", "strength": 0.9,
                "steps": 4, "guidance": 1.0, "width": 512, "height": 512,
                "request_id": "t-inp-add"}
        r = c.post("/v1/images/inpaint", json=body)
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        out = img_from(j["image"])
        assert out.size == src.size, f"size {out.size}"
        import numpy as np
        m = np.asarray(mask) > 127
        a = np.asarray(src, dtype=np.float32)
        b = np.asarray(out, dtype=np.float32)
        d_inside = float(abs(a[m] - b[m]).mean())
        d_outside = float(abs(a[~m] - b[~m]).mean())
        assert d_inside > 5, f"masked area unchanged ({d_inside:.2f})"
        assert d_outside < 2.0, f"unmasked area changed ({d_outside:.2f})"
        (OUT / "inpaint_add.png").write_bytes(base64.b64decode(j["image"]))
        return f"inside={d_inside:.1f} outside={d_outside:.2f} engine={j.get('engine')}"

    def inpaint_remove():
        body = {"image": b64(TD / "images" / "test_512.png"),
                "mask": b64(TD / "images" / "mask_512.png"),
                "prompt": "empty background", "mode": "remove", "strength": 0.9,
                "steps": 4, "request_id": "t-inp-rm"}
        r = c.post("/v1/images/inpaint", json=body)
        assert r.status_code == 200, r.text[:200]
        return f"engine={r.json().get('engine')}"

    def inpaint_color():
        src = Image.open(TD / "images" / "test_512.png").convert("RGB")
        body = {"image": b64(TD / "images" / "test_512.png"),
                "mask": b64(TD / "images" / "mask_512.png"),
                "prompt": "", "mode": "color", "color": "#0066ff",
                "request_id": "t-inp-col"}
        r = c.post("/v1/images/inpaint", json=body)
        assert r.status_code == 200, r.text[:200]
        out = img_from(r.json()["image"])
        assert diff(src, out) > 1.0, "color mode did not change the mask"
        return f"engine={r.json().get('engine')} diff={diff(src, out):.1f}"

    def outpaint():
        src = Image.open(TD / "images" / "test_512.png").convert("RGB")
        body = {"image": b64(TD / "images" / "test_512.png"),
                "prompt": "continue the scene, natural background",
                "margin": 0.25, "steps": 6, "request_id": "t-outpaint"}
        r = c.post("/v1/images/outpaint", json=body)
        assert r.status_code == 200, r.text[:300]
        j = r.json()
        out = img_from(j["image"])
        assert out.size == (768, 768), f"size {out.size}"
        assert j.get("output_size") == [768, 768], j.get("output_size")
        import numpy as np
        a = np.asarray(src, dtype=np.float32)
        b = np.asarray(out, dtype=np.float32)[128:640, 128:640]
        centre = float(abs(a - b).mean())
        assert centre < 2.0, f"original content changed (diff={centre:.2f})"
        border = np.concatenate([b[:0].ravel(), np.asarray(out, dtype=np.float32)[0:128, :].ravel()])
        assert float(border.std()) > 3, "border is flat/empty"
        (OUT / "outpaint_768.png").write_bytes(base64.b64decode(j["image"]))
        return (f"{j.get('input_size')} -> {j.get('output_size')} "
                f"centre_diff={centre:.2f} border_std={border.std():.1f}")

    def outpaint_validation():
        r = c.post("/v1/images/outpaint",
                   json={"image": b64(TD / "images" / "test_512.png"), "margin": 0})
        assert r.status_code == 400, f"expected 400 for no margins, got {r.status_code}"
        return r.json().get("error", "")[:60]

    def harmonize():
        r = c.post("/v1/images/harmonize",
                   json={"image": b64(TD / "images" / "test_512.png"),
                         "mask": b64(TD / "images" / "mask_512.png"),
                         "request_id": "t-harm"})
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        assert img_from(j["image"]).size == (512, 512)
        return f"engine={j.get('engine')}"

    check("generate 512x512", generate)
    check("generate custom 320x256", generate_custom)
    check("edit (img2img)", img2img)
    check("inpaint add", inpaint_add)
    check("inpaint remove", inpaint_remove)
    check("inpaint color (no AI)", inpaint_color)
    check("outpaint 512 -> 768", outpaint)
    check("outpaint rejects empty margins", outpaint_validation)
    check("harmonize", harmonize)

    # ---------------- upscale / vision ----------------
    def upscale():
        body = {"image": b64(TD / "images" / "test_512.png"), "scale": 2,
                "model": "RealESRGAN_x4v3", "request_id": "t-up"}
        r = c.post("/v1/images/upscale", json=body)
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        im = img_from(j["image"])
        assert im.size == (1024, 1024), f"size {im.size}"
        (OUT / "upscale_1024.png").write_bytes(base64.b64decode(j["image"]))
        return f"{im.size} engine={j.get('engine')}"

    def caption():
        body = {"image": b64(TD / "images" / "test_512.png"),
                "prompt": "Describe the image in one sentence.",
                "max_tokens": 64, "request_id": "t-cap"}
        r = c.post("/v1/images/caption", json=body)
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        text = (j.get("text") or j.get("caption") or "").strip()
        assert len(text) > 10, f"caption too short: {text!r}"
        return f"{len(text)} chars: {text[:60]}"

    def chat():
        body = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is in this image? One sentence."},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + b64(TD / "images" / "test_512.png")}}]}]}
        r = c.post("/v1/chat/completions", json=body)
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        text = j["choices"][0]["message"]["content"].strip()
        assert len(text) > 5, f"empty answer: {text!r}"
        return f"{len(text)} chars: {text[:60]}"

    check("upscale 512 -> 1024", upscale)
    check("caption (VLM)", caption)
    check("chat (vision)", chat)

    # ---------------- audio ----------------
    def transcribe():
        body = {"model": "small", "language": "en", "auto_split": "false"}
        files = {"file": ("jfk.wav", (TD / "audio" / "jfk.wav").read_bytes(), "audio/wav")}
        r = c.post("/v1/audio/transcriptions", files=files, data=body)
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        text = (j.get("text") or "").lower()
        assert "ask not what your country can do for you" in text, text[:120]
        return f"engine={j.get('engine')} lang={j.get('language')}: {text[:60]}"

    def transcribe_silence():
        body = {"model": "small", "language": "en", "auto_split": "false"}
        files = {"file": ("tone.wav", (TD / "audio" / "tone_3s.wav").read_bytes(), "audio/wav")}
        r = c.post("/v1/audio/transcriptions", files=files, data=body)
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        return f"text={j.get('text')!r} (silence handled, no crash)"

    def translations():
        files = {"file": ("jfk.wav", (TD / "audio" / "jfk.wav").read_bytes(), "audio/wav")}
        r = c.post("/v1/audio/translations", files=files, data={"model": "small"})
        assert r.status_code in (200, 503), r.text[:200]
        j = r.json()
        return f"{r.status_code} engine={j.get('engine')} text_len={len(j.get('text') or '')}"

    check("transcribe wav (jfk)", transcribe)
    check("transcribe silence", transcribe_silence)
    check("translations endpoint", translations)

    # ---------------- ffmpeg ----------------
    def presets():
        j = c.get("/v1/ffmpeg/presets").json()
        assert "video" in j and "audio" in j, list(j)
        return f"video={len(j['video'])} audio={len(j['audio'])} image={len(j.get('image', {}))}"

    def ff_convert_image():
        files = {"file": ("t.png", (TD / "images" / "test_512.png").read_bytes(), "image/png")}
        r = c.post("/v1/ffmpeg/convert-image", files=files, data={"preset": "jpg-95"})
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        raw = base64.b64decode(j.get("file") or j.get("data") or "")
        assert raw, "empty output"
        im = Image.open(io.BytesIO(raw))
        assert im.format == "JPEG", im.format
        return f"{im.size} {im.format} {len(raw)} B"

    def ff_convert_audio():
        files = {"file": ("jfk.wav", (TD / "audio" / "jfk.wav").read_bytes(), "audio/wav")}
        r = c.post("/v1/ffmpeg/convert-audio", files=files, data={"preset": "mp3-192"})
        assert r.status_code == 200, r.text[:200]
        raw = base64.b64decode(r.json().get("file") or "")
        assert raw[:3] == b"ID3" or raw[:2] == b"\xff\xfb", f"not mp3: {raw[:4]!r}"
        return f"{len(raw)} B"

    def ff_convert_video():
        files = {"file": ("t.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")}
        r = c.post("/v1/ffmpeg/convert-video", files=files, data={"preset": "mp4-h264"})
        assert r.status_code == 200, r.text[:200]
        raw = base64.b64decode(r.json().get("file") or "")
        p = OUT / "conv.mp4"
        p.write_bytes(raw)
        meta = ffprobe(p)
        codec = meta.get("streams", [{}])[0].get("codec_name", "?")
        return f"{len(raw)} B codec={codec}"

    def ff_extract_audio():
        files = {"file": ("t.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")}
        r = c.post("/v1/ffmpeg/extract-audio", files=files, data={"preset": "mp3-192"})
        assert r.status_code == 200, r.text[:200]
        raw = base64.b64decode(r.json().get("file") or "")
        assert len(raw) > 1000
        return f"{len(raw)} B"

    def ff_trim():
        files = {"file": ("t.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")}
        r = c.post("/v1/ffmpeg/trim", files=files,
                   data={"start": "1", "end": "3", "preset": "copy"})
        assert r.status_code == 200, r.text[:200]
        raw = base64.b64decode(r.json().get("file") or "")
        p = OUT / "trim.mp4"
        p.write_bytes(raw)
        meta = ffprobe(p)
        dur = float(meta.get("format", {}).get("duration", 0))
        assert 1.5 <= dur <= 3.5, f"trimmed duration {dur}"
        return f"{dur:.2f}s {len(raw)} B"

    def ff_compress():
        files = {"file": ("t.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")}
        r = c.post("/v1/ffmpeg/compress", files=files, data={"preset": "web-720p"})
        assert r.status_code == 200, r.text[:200]
        raw = base64.b64decode(r.json().get("file") or "")
        assert len(raw) > 1000
        return f"{len(raw)} B"

    def ff_info():
        files = {"file": ("t.mp4", (TD / "video" / "test_5s.mp4").read_bytes(), "video/mp4")}
        r = c.post("/v1/ffmpeg/info", files=files)
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        meta = j.get("info") or j
        assert "streams" in json.dumps(meta), "no streams in info"
        return f"keys={list(j)[:5]}"

    check("ffmpeg presets", presets)
    check("ffmpeg convert-image", ff_convert_image)
    check("ffmpeg convert-audio", ff_convert_audio)
    check("ffmpeg convert-video", ff_convert_video)
    check("ffmpeg extract-audio", ff_extract_audio)
    check("ffmpeg trim 1s..3s", ff_trim)
    check("ffmpeg compress", ff_compress)
    check("ffmpeg info", ff_info)

    # ---------------- model management ----------------
    def model_status():
        j = c.get("/v1/models/status/RealESRGAN_x4v3").json()
        assert j.get("installed") is True, j
        return f"installed={j['installed']} kind={j.get('kind')}"

    def model_verify():
        j = c.post("/v1/models/verify/RealESRGAN_x4v3").json()
        assert j.get("verified") is True, j
        return f"verified={j['verified']} ({j.get('method','')[:40]})"

    def model_progress():
        j = c.get("/v1/models/progress/small").json()
        assert j.get("state") == "done", j
        return f"state={j['state']} pct={j['percent']} bytes={j['bytes_total']}"

    def model_unknown():
        r = c.get("/v1/models/status/no-such-model")
        assert r.status_code == 404, r.status_code
        return r.json().get("error", "")[:40]

    check("models status (upscale)", model_status)
    check("models verify", model_verify)
    check("models progress (whisper)", model_progress)
    check("models unknown id -> 404", model_unknown)

    # ---------------- extra: sizes, errors, concurrency ----------------
    def upscale4():
        body = {"image": b64(TD / "images" / "test_512.png"), "scale": 4,
                "model": "RealESRGAN_x4v3", "request_id": "t-up4"}
        r = c.post("/v1/images/upscale", json=body)
        assert r.status_code == 200, r.text[:200]
        im = img_from(r.json()["image"])
        assert im.size == (2048, 2048), f"size {im.size}"
        return f"{im.size}"

    def gen_n2():
        body = {"prompt": "a lighthouse at dusk", "size": "256x256", "n": 2}
        r = c.post("/v1/images/generations", json=body)
        assert r.status_code == 200, r.text[:200]
        j = r.json()
        assert len(j["data"]) == 2, f"got {len(j['data'])} images"
        a = j["data"][0]["b64_json"]
        b = j["data"][1]["b64_json"]
        assert a != b, "n=2 returned the same image twice"
        return f"2 distinct images, size={j.get('size')}"

    def err_missing_image():
        r = c.post("/v1/images/inpaint", json={"prompt": "x"})
        assert r.status_code == 400, f"expected 400, got {r.status_code}"
        return r.json().get("error", "")[:50]

    def err_bad_b64():
        r = c.post("/v1/images/upscale", json={"image": "not-base64!!", "scale": 2})
        assert r.status_code in (400, 500), f"expected an error, got {r.status_code}"
        return f"{r.status_code}: {r.text[:60]}"

    def err_outpaint_too_big():
        r = c.post("/v1/images/outpaint",
                   json={"image": b64(TD / "images" / "test_512.png"),
                         "left": 5000, "right": 5000, "prompt": "x"})
        assert r.status_code == 400, f"expected 400, got {r.status_code}"
        return r.json().get("error", "")[:70]

    def concurrency():
        import concurrent.futures
        def one(i):
            return c.post("/v1/images/generations",
                          json={"prompt": f"abstract shape {i}", "size": "256x256",
                                "request_id": f"t-conc-{i}"}).status_code
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            codes = list(ex.map(one, (1, 2)))
        assert codes == [200, 200], f"codes {codes}"
        h = c.get("/health").json()
        assert h.get("busy") in (True, False)
        return f"both 200, busy={h.get('busy')} queue={h.get('queue')}"

    def transcribe_turbo():
        models = {m["id"]: m for m in c.get("/v1/models").json()["data"]}
        if not models.get("turbo", {}).get("installed"):
            return "skipped (turbo not installed)"
        files = {"file": ("jfk.wav", (TD / "audio" / "jfk.wav").read_bytes(), "audio/wav")}
        r = c.post("/v1/audio/transcriptions", files=files,
                   data={"model": "turbo", "language": "en", "auto_split": "false"})
        assert r.status_code == 200, r.text[:200]
        text = (r.json().get("text") or "").lower()
        assert "ask not what your country can do for you" in text, text[:120]
        return f"engine={r.json().get('engine')}: {text[:50]}"

    check("upscale 4x (512 -> 2048)", upscale4)
    check("generate n=2 distinct", gen_n2)
    check("error: missing image -> 400", err_missing_image)
    check("error: invalid base64", err_bad_b64)
    check("error: outpaint too large -> 400", err_outpaint_too_big)
    check("concurrency: 2 parallel generations", concurrency)
    check("transcribe with turbo", transcribe_turbo)

    # ---------------- summary ----------------
    ok = sum(1 for r in results if r["ok"])
    total = len(results)
    print(f"\n{ok}/{total} passed")
    failed = [r for r in results if not r["ok"]]
    if failed:
        print("FAILURES:")
        for r in failed:
            print(f"  - {r['name']}: {r['detail']}")
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"artifacts: {OUT}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
