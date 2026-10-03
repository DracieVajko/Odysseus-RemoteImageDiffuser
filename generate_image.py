#!/usr/bin/env python3
"""CLI for DiffusionServer: generate without a browser.

Examples:
  python generate_image.py health
  python generate_image.py models
  python generate_image.py doctor
  python generate_image.py txt2img --model DreamShaper_8 --prompt "a cabin" --output out.png
  python generate_image.py img2img --model DreamShaper_8 --prompt "snow" --input ref.png --output out.png --strength 0.75
  python generate_image.py inpaint --model DreamShaper_8 --prompt "a door" --input img.png --mask mask.png --output out.png
"""
import argparse
import base64
import io
import json
import sys
import urllib.error
import urllib.request

SERVER = "http://127.0.0.1:8101"


def req_json(path, payload, timeout=600):
    r = urllib.request.Request(SERVER + path, data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"raw": e.read().decode()[:300]}


def req_multipart(path, fields, files, timeout=600):
    bound = "----diffusioncli"
    body = b""
    for k, v in fields.items():
        body += f'--{bound}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (fname, data, ctype) in files.items():
        body += (f'--{bound}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{bound}--\r\n".encode()
    r = urllib.request.Request(SERVER + path, data=body,
                               headers={"Content-Type": f"multipart/form-data; boundary={bound}"},
                               method="POST")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"raw": e.read().decode()[:300]}


def save_b64(b64, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"saved -> {path}")


def cmd_health(_):
    with urllib.request.urlopen(SERVER + "/health", timeout=15) as r:
        print(json.dumps(json.loads(r.read().decode()), indent=1))


def cmd_models(_):
    with urllib.request.urlopen(SERVER + "/v1/models", timeout=15) as r:
        m = json.loads(r.read().decode())
    for d in m.get("data", []):
        flag = "" if d.get("available", True) else " (MISSING FILES)"
        print(f"- {d['id']}{flag}")
    if m.get("missing_models"):
        print("missing:", ", ".join(m["missing_models"]))
    print("loaded:", m.get("currently_loaded_model"))


def cmd_doctor(_):
    import os
    from pathlib import Path
    here = Path(__file__).resolve().parent
    cfg = json.loads((here / "models.json").read_text(encoding="utf-8"))
    ok = True
    print(f"port: {cfg.get('port')}  default: {cfg.get('default_model')}")
    for mid, prof in cfg.get("models", {}).items():
        p = prof.get("path", "")
        exists = Path(p).exists() if p else False
        ok &= exists
        print(f"[{'OK' if exists else 'MISSING'}] {mid} -> {p}")
    for name in ("webui.html", "diffusion_server.py", "download_models.py"):
        e = (here / name).exists()
        ok &= e
        print(f"[{'OK' if e else 'MISSING'}] {name}")
    try:
        with urllib.request.urlopen(SERVER + "/health", timeout=10) as r:
            print("server:", r.status, json.loads(r.read().decode()).get("status"))
    except Exception as e:  # noqa: BLE001
        ok = False
        print("server: UNREACHABLE", str(e)[:120])
    try:
        import torch
        print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), end="")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f" vram_free={free // 1048576}MiB total={total // 1048576}MiB", end="")
        print()
    except Exception as e:  # noqa: BLE001
        print("torch: n/a", str(e)[:100])
    sys.exit(0 if ok else 1)


def cmd_txt2img(a):
    st, b = req_json("/v1/images/generations",
                     {"model": a.model, "prompt": a.prompt, "size": a.size,
                      "quality": a.quality, "n": 1})
    if st != 200:
        print(f"Error {st}: {json.dumps(b)[:400]}")
        sys.exit(1)
    save_b64(b["data"][0]["b64_json"], a.output)


def cmd_img2img(a):
    with open(a.input, "rb") as f:
        raw = f.read()
    st, b = req_multipart("/v1/images/edits",
                          {"prompt": a.prompt, "model": a.model, "size": a.size,
                           "quality": a.quality, "strength": str(a.strength)},
                          {"image": ("input.png", raw, "image/png")})
    if st != 200:
        print(f"Error {st}: {json.dumps(b)[:400]}")
        sys.exit(1)
    save_b64(b["data"][0]["b64_json"], a.output)


def cmd_inpaint(a):
    def b64(path):
        from PIL import Image
        buf = io.BytesIO()
        Image.open(path).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    st, b = req_json("/v1/images/inpaint",
                     {"image": b64(a.input), "mask": b64(a.mask), "prompt": a.prompt,
                      "model": a.model, "strength": a.strength, "steps": a.steps})
    if st != 200:
        print(f"Error {st}: {json.dumps(b)[:400]}")
        sys.exit(1)
    save_b64(b["image"], a.output)


def main():
    ap = argparse.ArgumentParser(description="DiffusionServer CLI")
    ap.add_argument("--server", default="http://127.0.0.1:8101")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("health")
    sub.add_parser("models")
    sub.add_parser("doctor")

    p = sub.add_parser("txt2img")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--size", default="768x768")
    p.add_argument("--quality", default="medium")
    p.add_argument("--output", required=True)

    p = sub.add_parser("img2img")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--size", default="768x768")
    p.add_argument("--quality", default="medium")
    p.add_argument("--strength", type=float, default=0.75)
    p.add_argument("--output", required=True)

    p = sub.add_parser("inpaint")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--mask", required=True)
    p.add_argument("--strength", type=float, default=0.75)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--output", required=True)

    a = ap.parse_args()
    global SERVER
    SERVER = a.server.rstrip("/")
    {"health": cmd_health, "models": cmd_models, "doctor": cmd_doctor,
     "txt2img": cmd_txt2img, "img2img": cmd_img2img, "inpaint": cmd_inpaint}[a.cmd](a)


if __name__ == "__main__":
    main()
