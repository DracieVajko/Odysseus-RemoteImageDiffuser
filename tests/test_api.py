"""Full API test for the Odysseus diffusion helper.
Usage (venv python):
  python tests\\test_api.py --model DreamShaper [--remote-host 100.88.219.120]
Does: /health, /v1/models, generations (saves PNG), edits, inpaint, outpaint-via-inpaint.
Measures: VRAM (nvidia-smi), RAM, generation time. Prints JSON summary.
"""
import argparse, base64, io, json, os, subprocess, sys, time, urllib.request, urllib.error

SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SERVER_DIR)

def http_json(url, data=None, timeout=600):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def http_get(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())

def nvidia_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip().splitlines()[0].split(",")
        return {"vram_used_mib": int(out[0].strip()), "vram_total_mib": int(out[1].strip()), "gpu_util_pct": int(out[2].strip())}
    except Exception as e:
        return {"vram_error": str(e)}

def ram_mem():
    try:
        import ctypes
        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = MS(); ms.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return {"ram_total_mib": int(ms.ullTotalPhys/1048576), "ram_avail_mib": int(ms.ullAvailPhys/1048576),
                "ram_used_mib": int((ms.ullTotalPhys-ms.ullAvailPhys)/1048576)}
    except Exception as e:
        return {"ram_error": str(e)}

def save_b64_png(b64, path):
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["DreamShaper_8","RealisticVision_V6_B1","MeinaMix_V11",
                             "DreamShaper","RealisticVision","MeinaMix"])
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    _ALIAS = {"DreamShaper": "DreamShaper_8", "RealisticVision": "RealisticVision_V6_B1", "MeinaMix": "MeinaMix_V11"}
    args.model = _ALIAS.get(args.model, args.model)

    with open(os.path.join(SERVER_DIR, "models.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    prof = cfg["models"][args.model]
    base = f"http://{args.host}:{args.port}"
    outdir = os.path.join(SERVER_DIR, "tests", "output")
    os.makedirs(outdir, exist_ok=True)
    neg = cfg.get("negative_prompt", "")
    summary = {"model": args.model, "profile": prof["label"]}

    print(f"== {args.model}: /health ==", flush=True)
    summary["health"] = http_get(base + "/health")
    print(summary["health"], flush=True)
    print(f"== /v1/models ==", flush=True)
    summary["models"] = http_get(base + "/v1/models")
    print(summary["models"], flush=True)
    summary["vram_idle"] = nvidia_mem()
    summary["ram_idle"] = ram_mem()
    print("idle:", summary["vram_idle"], summary["ram_idle"], flush=True)

    # --- TEST: txt2img ---
    from PIL import Image
    print(f"== generations ({prof['test_size']}, {prof['steps']} steps) ==", flush=True)
    t0 = time.time()
    gen = http_json(base + "/v1/images/generations", {
        "model": args.model,
        "prompt": prof["test_prompt"],
        "size": prof["test_size"],
        "n": 1,
        # quality maps to server default steps; we pass explicit size, steps come from --steps launch arg.
    }, timeout=600)
    dt_gen = round(time.time() - t0, 1)
    summary["vram_peak"] = nvidia_mem()
    summary["ram_peak"] = ram_mem()
    summary["gen_seconds"] = dt_gen
    b64 = gen["data"][0]["b64_json"]
    gen_path = os.path.join(outdir, f"{args.model}-txt2img.png")
    save_b64_png(b64, gen_path)
    img = Image.open(gen_path)
    summary["gen_png"] = gen_path
    summary["gen_size_px"] = list(img.size)
    summary["gen_bytes"] = os.path.getsize(gen_path)
    # quality sanity: not all-grey/black, has variance
    import numpy as np
    arr = np.asarray(img.convert("RGB")).astype(float)
    summary["gen_pixel_mean"] = round(float(arr.mean()), 1)
    summary["gen_pixel_std"] = round(float(arr.std()), 1)
    print(f"saved {gen_path} {img.size} in {dt_gen}s mean={summary['gen_pixel_mean']} std={summary['gen_pixel_std']}", flush=True)

    # --- TEST: edits (multipart, like OpenAI) ---
    print("== edits ==", flush=True)
    import urllib.parse, mimetypes
    boundary = "----testboundary123"
    with open(gen_path, "rb") as f:
        png_bytes = f.read()
    fields = {"prompt": "make it slightly more vibrant, same composition", "model": args.model, "size": prof["test_size"]}
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"input.png\"\r\nContent-Type: image/png\r\n\r\n".encode())
    body.write(png_bytes + f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(base + "/v1/images/edits", data=body.getvalue(),
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        edit_res = json.loads(r.read().decode())
    summary["edit_seconds"] = round(time.time() - t0, 1)
    edit_path = os.path.join(outdir, f"{args.model}-edit.png")
    save_b64_png(edit_res["data"][0]["b64_json"], edit_path)
    summary["edit_png"] = edit_path
    summary["edit_bytes"] = os.path.getsize(edit_path)
    print(f"saved {edit_path} in {summary['edit_seconds']}s", flush=True)

    # --- TEST: inpaint (center mask) ---
    print("== inpaint ==", flush=True)
    from PIL import ImageDraw
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.rectangle([w//4, h//4, 3*w//4, 3*h//4], fill=255)
    buf_i, buf_m = io.BytesIO(), io.BytesIO()
    img.save(buf_i, format="PNG"); mask.save(buf_m, format="PNG")
    t0 = time.time()
    inp = http_json(base + "/v1/images/inpaint", {
        "model": args.model,
        "image": base64.b64encode(buf_i.getvalue()).decode(),
        "mask": base64.b64encode(buf_m.getvalue()).decode(),
        "prompt": "a beautiful garden fountain, detailed",
        "width": w, "height": h,
    }, timeout=600)
    summary["inpaint_seconds"] = round(time.time() - t0, 1)
    inpaint_path = os.path.join(outdir, f"{args.model}-inpaint.png")
    save_b64_png(inp["image"], inpaint_path)
    summary["inpaint_png"] = inpaint_path
    summary["inpaint_elapsed_server"] = inp.get("elapsed")
    print(f"saved {inpaint_path} in {summary['inpaint_seconds']}s", flush=True)

    # --- TEST: outpaint via inpaint (Odysseus flow: expanded canvas + mask) ---
    # No dedicated /outpaint endpoint exists: expand canvas to the right by 50%,
    # paste original on the left, mask the new region, call /v1/images/inpaint.
    print("== outpaint-via-inpaint ==", flush=True)
    new_w = w + w // 2
    canvas = Image.new("RGB", (new_w, h), (128, 128, 128))
    canvas.paste(img, (0, 0))
    omask = Image.new("L", (new_w, h), 0)
    ImageDraw.Draw(omask).rectangle([w, 0, new_w, h], fill=255)
    buf_c, buf_o = io.BytesIO(), io.BytesIO()
    canvas.save(buf_c, format="PNG"); omask.save(buf_o, format="PNG")
    t0 = time.time()
    outp = http_json(base + "/v1/images/inpaint", {
        "model": args.model,
        "image": base64.b64encode(buf_c.getvalue()).decode(),
        "mask": base64.b64encode(buf_o.getvalue()).decode(),
        "prompt": prof["test_prompt"] + ", seamless continuation",
        "width": new_w, "height": h,
    }, timeout=600)
    summary["outpaint_seconds"] = round(time.time() - t0, 1)
    outpaint_path = os.path.join(outdir, f"{args.model}-outpaint.png")
    save_b64_png(outp["image"], outpaint_path)
    oimg = Image.open(outpaint_path)
    summary["outpaint_png"] = outpaint_path
    summary["outpaint_size_px"] = list(oimg.size)
    print(f"saved {outpaint_path} {oimg.size} in {summary['outpaint_seconds']}s", flush=True)

    summary["vram_end"] = nvidia_mem()
    summary["ram_end"] = ram_mem()
    spath = os.path.join(outdir, f"{args.model}-summary.json")
    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY " + json.dumps(summary, indent=2), flush=True)
    print(f"wrote {spath}", flush=True)

if __name__ == "__main__":
    main()
