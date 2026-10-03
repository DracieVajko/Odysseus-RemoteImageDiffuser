#!/usr/bin/env python3
"""Visibility contract test: progress / health-busy / 503 / 400s / rid echo.

Run AFTER production restart (fresh, no model loaded).
Leaves DreamShaper_8 resident at the end.
"""
import base64
import io
import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8101"
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}", flush=True)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        return r.status, json.loads(r.read().decode())


def post_json(path, payload, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}


def post_multipart(path, fields, files, timeout=600):
    bound = "----vistest999"
    body = b""
    for k, v in fields.items():
        body += f'--{bound}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    for k, (fname, data, ctype) in files.items():
        body += (f'--{bound}\r\nContent-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'
                 f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{bound}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={bound}"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {}


def progress(rid):
    try:
        return get(f"/v1/images/progress/{rid}")[1]
    except Exception:
        return {}


def main():
    # 1) health shape on fresh server
    _, h = get("/health")
    check("health-busy-fields", all(k in h for k in ("busy", "queue", "current_model")),
          str({k: h.get(k) for k in ("busy", "queue", "current_model", "model")}))
    check("health-idle", h.get("busy") is False and h.get("queue") == 0)

    # 2) 503 during cold load: RV loads (slow) in background, DS asked meanwhile
    res_a, prog_seen = {}, []
    rid_a = "vis-load-rv-001"

    def load_rv():
        res_a["r"] = post_json("/v1/images/generations",
                               {"model": "RealisticVision_V6_B1", "prompt": "a cabin",
                                "size": "512x512", "quality": "low", "n": 1, "request_id": rid_a})

    t = threading.Thread(target=load_rv, daemon=True)
    t.start()
    time.sleep(20)
    _, h2 = get("/health")
    check("health-loading", h2.get("loading") is True, str(h2.get("loading_model")))
    st_b, body_b = post_json("/v1/images/generations",
                             {"model": "DreamShaper_8", "prompt": "x", "size": "512x512",
                              "quality": "low", "n": 1, "request_id": "vis-probe-503"})
    check("503-during-load", st_b == 503 and body_b.get("error") == "loading model",
          f"status={st_b} body={str(body_b)[:120]}")
    p1 = progress(rid_a)
    check("progress-during-load", p1.get("status") in ("loading", "running"), str(p1))
    t.join(timeout=590)
    st_a, body_a = res_a.get("r", (0, {}))
    check("load-completes-200", st_a == 200, f"status={st_a}")
    check("rid-echo-generations", body_a.get("request_id") == rid_a, str(body_a.get("request_id")))
    p2 = progress(rid_a)
    check("progress-done", p2.get("status") == "done" and p2.get("percent") == 100, str(p2))

    # 3) 400s (model warm now -> fast, no load involved)
    st, b = post_json("/v1/images/generations",
                      {"model": "RealisticVision_V6_B1", "prompt": "x", "size": "1x1",
                       "quality": "low", "n": 1})
    check("400-tiny-size", st == 400, f"status={st} {str(b)[:100]}")
    st, b = post_json("/v1/images/generations",
                      {"model": "RealisticVision_V6_B1", "prompt": "x", "size": "512x512",
                       "quality": "low", "n": 99})
    check("400-bad-n", st == 400, f"status={st}")
    st, b = post_json("/v1/images/inpaint",
                      {"image": "!!!not-base64!!!", "mask": "!!!", "prompt": "x", "model": ""})
    check("400-bad-b64", st == 400, f"status={st} {str(b)[:100]}")
    st, b = post_multipart("/v1/images/edits",
                           {"prompt": "x", "model": "RealisticVision_V6_B1", "size": "512x512",
                            "quality": "low", "strength": "0.75"},
                           {"image": ("evil.txt", b"this is not an image", "text/plain")})
    check("400-bad-upload", st == 400, f"status={st} {str(b)[:100]}")

    # 4) inpaint valid + progress + echo
    from PIL import Image
    img = Image.new("RGB", (256, 256), (120, 30, 30))
    msk = Image.new("L", (256, 256), 0)
    px = msk.load()
    for y in range(96, 160):
        for x in range(96, 160):
            px[x, y] = 255
    bi, bm = io.BytesIO(), io.BytesIO()
    img.save(bi, format="PNG")
    msk.save(bm, format="PNG")
    rid_i = "vis-inpaint-001"
    st, b = post_json("/v1/images/inpaint",
                      {"image": base64.b64encode(bi.getvalue()).decode(),
                       "mask": base64.b64encode(bm.getvalue()).decode(),
                       "prompt": "blue square", "model": "RealisticVision_V6_B1",
                       "steps": 4, "strength": 0.8, "request_id": rid_i}, timeout=300)
    check("inpaint-200", st == 200, f"status={st} {str(b)[:100]}")
    check("rid-echo-inpaint", b.get("request_id") == rid_i)
    check("progress-inpaint-done", progress(rid_i).get("status") == "done")

    # 5) harmonize Reinhard-only (model-free) + echo
    rid_h = "vis-harm-001"
    st, b = post_json("/v1/images/harmonize",
                      {"image": base64.b64encode(bi.getvalue()).decode(), "prompt": "x",
                       "color_match": 0.5, "seam_fix": 0.0,
                       "body_mask": base64.b64encode(bm.getvalue()).decode(),
                       "request_id": rid_h}, timeout=120)
    check("harmonize-200", st == 200, f"status={st} {str(b)[:100]}")
    check("rid-echo-harmonize", b.get("request_id") == rid_h)
    check("progress-harmonize-done", progress(rid_h).get("status") == "done")

    # 6) back to DreamShaper (leave resident as before)
    st, b = post_json("/v1/images/generations",
                      {"model": "DreamShaper_8", "prompt": "a cabin",
                       "size": "512x512", "quality": "low", "n": 1,
                       "request_id": "vis-back-ds"})
    check("switchback-200", st == 200, f"status={st}")

    print(f"\nTOTAL PASS={len(PASS)} FAIL={len(FAIL)}", flush=True)
    if FAIL:
        print("FAILED:", FAIL, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
