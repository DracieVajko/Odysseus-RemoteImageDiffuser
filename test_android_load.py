import base64
import os
import subprocess
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8199"
ADB = os.path.join(os.environ["LOCALAPPDATA"], "Android", "Sdk", "platform-tools", "adb.exe")
TD = Path(r"D:\AI\DiffusionServer\testdata")

c = httpx.Client(base_url=BASE, timeout=1200.0)
img_b64 = base64.b64encode((TD / "images" / "test_512.png").read_bytes()).decode()
mask_b64 = base64.b64encode((TD / "images" / "mask_512.png").read_bytes()).decode()


def adb(*args):
    return subprocess.run([ADB, *args], capture_output=True, text=True, timeout=30).stdout.strip()


def snapshot(label):
    pid = adb("shell", "pidof", "eu.odysseus.diffusion")
    mem = ""
    if pid:
        out = adb("shell", f"cat /proc/{pid}/status 2>/dev/null")
        for line in out.splitlines():
            if line.startswith(("VmRSS", "Threads")):
                mem += " " + line.strip()
    print(f"[{label}] pid={pid or 'NONE'}{mem}", flush=True)


def post(name, path, **kw):
    t0 = time.time()
    try:
        r = c.request("POST", path, **kw)
        print(f"  {name:<18} {r.status_code} {time.time()-t0:6.1f}s", flush=True)
        return r.status_code
    except Exception as e:
        print(f"  {name:<18} EXC {e}", flush=True)
        return 0


snapshot("start")
counts = {"ok": 0, "fail": 0}
for i in range(20):
    if i % 3 == 0:
        code = post(f"generate {i}", "/v1/images/generations", json={
            "prompt": "a red cube", "negative_prompt": "blurry",
            "guidance": 1.0, "size": "256x256", "n": 1, "steps": 4})
    elif i % 3 == 1:
        code = post(f"inpaint color {i}", "/v1/images/inpaint", json={
            "image": img_b64, "mask": mask_b64, "prompt": "",
            "width": 256, "height": 256, "mode": "color", "color": "#0066ff"})
    else:
        code = post(f"harmonize {i}", "/v1/images/harmonize", json={
            "image": img_b64, "mask": mask_b64, "strength": 0.6})
    if code == 200:
        counts["ok"] += 1
    else:
        counts["fail"] += 1
    if i + 1 in (5, 10, 15, 20):
        snapshot(f"after {i+1}")

snapshot("end")
print("summary:", counts)
