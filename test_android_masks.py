import base64
import io
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

BASE = "http://127.0.0.1:8199"
TD = Path(r"D:\AI\DiffusionServer\testdata")
c = httpx.Client(base_url=BASE, timeout=1200.0)
src_bytes = (TD / "images" / "test_512.png").read_bytes()
src_b64 = base64.b64encode(src_bytes).decode()
src = np.asarray(Image.open(io.BytesIO(src_bytes)).convert("RGB")).astype(np.int16)

masks = {
    "black": Image.new("L", (512, 512), 0),
    "white": Image.new("L", (512, 512), 255),
    "gray128": Image.new("L", (512, 512), 128),
}


def alpha_mask():
    img = Image.new("RGBA", (512, 512), (255, 255, 255, 0))
    arr = np.asarray(img).copy()
    arr[..., 3] = 128  # semi-transparent white everywhere
    return Image.fromarray(arr, "RGBA")


masks["alpha128"] = alpha_mask()

for name, mask in masks.items():
    buf = io.BytesIO()
    mask.save(buf, "PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode()
    r = c.post("/v1/images/inpaint", json={
        "image": src_b64, "mask": mask_b64, "prompt": "a glowing blue sphere",
        "strength": 0.9, "width": 256, "height": 256, "steps": 4, "mode": "add",
    })
    if r.status_code != 200:
        print(name, r.status_code, r.text[:120])
        continue
    out = np.asarray(Image.open(io.BytesIO(base64.b64decode(r.json()["image"]))).convert("RGB").resize((512, 512))).astype(np.int16)
    d = np.abs(out - src).mean()
    print(f"{name:<10} 200  mean_abs_diff={d:.1f}")
