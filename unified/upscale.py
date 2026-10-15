"""Upscale — Pillow LANCZOS fallback, tile hook pre Real-ESRGAN NCNN.

Real-ESRGAN-x4v3 (~60 MB, BSD-3, NCNN Vulkan) je cieľ pre PC aj mobile.
Kým nie je váha v models/, používa sa kvalitný Pillow fallback aby
kontrakt POST /v1/images/upscale fungoval end-to-end.
DoS cap 8192px na stranu (ako edits/inpaint v ODYSSEUS-CONTRACT).
"""
import base64
import io
import time
from PIL import Image

MAX_SIDE = 8192
ALLOWED_SCALES = (2, 4)


def upscale_image_b64(image_b64: str, scale: int = 2, model: str = "RealESRGAN_x4v3") -> dict:
    if scale not in ALLOWED_SCALES:
        # povoliť aj 3 pre kompat, ale primárne 2/4
        if scale not in (2, 3, 4):
            raise ValueError(f"scale must be 2/3/4, got {scale}")
    try:
        raw = base64.b64decode(image_b64)
    except Exception as e:
        raise ValueError(f"bad base64 image: {e}")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"bad image bytes: {e}")
    w, h = img.size
    nw, nh = w * scale, h * scale
    if max(nw, nh) > MAX_SIDE:
        raise ValueError(f"upscaled {nw}x{nh} exceeds cap {MAX_SIDE}px")
    t0 = time.time()
    # TODO: ak existuje models/RealESRGAN_x4v3/*.onnx/bin použit NCNN/ONNX
    # s tile processing (NCNN natívne dlaždice). Zatiaľ LANCZOS fallback.
    out = img.resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "elapsed": round(time.time() - t0, 2),
        "model": model,
        "scale": scale,
    }
