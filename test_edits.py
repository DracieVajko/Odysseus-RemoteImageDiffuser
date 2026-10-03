#!/usr/bin/env python3
"""Test /v1/images/edits (multipart img2img): expect 200 + real PNG."""
import io
import json
import sys
import time
import urllib.error
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8101"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "DreamShaper_8"
BASE = f"http://127.0.0.1:{PORT}"
OUT = rf"D:\AI\DiffusionServer\edittest-{PORT}-{MODEL}.png"

BOUND = "----testboundary1234"


def multipart(fields, files):
    body = b""
    for k, v in fields.items():
        body += (f"--{BOUND}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    for k, (fname, data, ctype) in files.items():
        body += (f"--{BOUND}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{BOUND}--\r\n".encode()
    return body


def main():
    from PIL import Image
    img = Image.new("RGB", (512, 512), (90, 140, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    fields = {"prompt": "same abstract shapes, sunset colors, painterly",
              "model": MODEL, "size": "512x512", "quality": "low",
              "strength": "0.75", "response_format": "b64_json", "request_id": ""}
    data = multipart(fields, {"image": ("ref.png", buf.getvalue(), "image/png")})
    req = urllib.request.Request(BASE + "/v1/images/edits", data=data,
                                 headers={"Content-Type": f"multipart/form-data; boundary={BOUND}"},
                                 method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            status, payload = r.status, json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001 - want body on HTTPError too
        if isinstance(e, urllib.error.HTTPError):
            print(f"status={e.code} elapsed={time.time()-t0:.1f}s", flush=True)
            print("BODY:", e.read().decode()[:600], flush=True)
            sys.exit(1)
        raise
    elapsed = time.time() - t0
    items = payload.get("data", [])
    print(f"status={status} elapsed={elapsed:.1f}s images={len(items)}", flush=True)
    if not items:
        print("BODY:", json.dumps(payload)[:500], flush=True)
        sys.exit(1)
    import base64
    raw = base64.b64decode(items[0]["b64_json"])
    print(f"bytes={len(raw)} is_png={raw[:8] == bytes.fromhex('89504e470d0a1a0a')}", flush=True)
    with open(OUT, "wb") as f:
        f.write(raw)
    print(f"saved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
