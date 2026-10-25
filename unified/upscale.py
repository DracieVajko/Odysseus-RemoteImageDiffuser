"""Upscale — Real-ESRGAN-x4v3 (RRDBNet) when weights + torch exist,
else Real-ESRGAN ONNX when available, else Pillow LANCZOS.

Weights: models/RealESRGAN_x4v3/RealESRGAN_x4plus.pth (BSD-3-Clause, 67 MB,
`download_models.py --preset upscale`). ONNX: models/RealESRGAN_x4v3/RealESRGAN_x4plus.onnx.
Tiled inference (tile 256, overlap 16) keeps VRAM low — same pattern as NCNN tiles on mobile.
DoS cap 8192px per side. scale 2/3: ESRGAN is x4 → the result is downsampled.
"""
import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger("unified.upscale")

MAX_SIDE = 8192
WEIGHT_NAMES = ("RealESRGAN_x4plus.pth", "RealESRGAN_x4plus.pt")

_HERE = Path(__file__).resolve().parent.parent
_DEFAULT_DIRS = (
    _HERE / "models" / "RealESRGAN_x4v3",
    Path("D:/AI/HFCache"),
)


def find_weight(dirs=None):
    for d in (dirs if dirs is not None else _DEFAULT_DIRS):
        d = Path(d)
        for name in WEIGHT_NAMES:
            p = d / name
            if p.is_file() and p.stat().st_size > 10 * 1024 * 1024:
                return p
    return None


def find_onnx_weight(dirs=None):
    for d in (dirs if dirs is not None else _DEFAULT_DIRS):
        d = Path(d)
        p = d / "RealESRGAN_x4plus.onnx"
        if p.is_file():
            data = p.with_suffix(".onnx.data")
            size = p.stat().st_size + (data.stat().st_size if data.is_file() else 0)
            if size > 10 * 1024 * 1024:
                return p
    return None


# ---------- vendored RRDBNet (x4plus: 64 feat, 23 blocks) ----------

def _rrdb_net():
    import torch
    from torch import nn
    import torch.nn.functional as F

    def _conv(in_ch, out_ch, k=3, s=1):
        return nn.Conv2d(in_ch, out_ch, k, s, k // 2)

    class ResidualDenseBlock(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.conv1 = _conv(nf, gc)
            self.conv2 = _conv(nf + gc, gc)
            self.conv3 = _conv(nf + 2 * gc, gc)
            self.conv4 = _conv(nf + 3 * gc, gc)
            self.conv5 = _conv(nf + 4 * gc, nf)

        def forward(self, x):
            x1 = F.leaky_relu(self.conv1(x), 0.2, True)
            x2 = F.leaky_relu(self.conv2(torch.cat((x, x1), 1)), 0.2, True)
            x3 = F.leaky_relu(self.conv3(torch.cat((x, x1, x2), 1)), 0.2, True)
            x4 = F.leaky_relu(self.conv4(torch.cat((x, x1, x2, x3), 1)), 0.2, True)
            return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(nf, gc)
            self.rdb2 = ResidualDenseBlock(nf, gc)
            self.rdb3 = ResidualDenseBlock(nf, gc)

        def forward(self, x):
            # BasicSR RRDB: sequential dense blocks + 0.2 residual scaling.
            out = self.rdb2(self.rdb1(x))
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        """BasicSR-compatible RRDBNet (official RealESRGAN_x4plus key names)."""

        def __init__(self):
            super().__init__()
            self.conv_first = _conv(3, 64)
            self.body = nn.Sequential(*[RRDB(64) for _ in range(23)])
            self.conv_body = _conv(64, 64)
            self.conv_up1 = _conv(64, 64)
            self.conv_up2 = _conv(64, 64)
            self.conv_hr = _conv(64, 64)
            self.conv_last = _conv(64, 3)

        def forward(self, x):
            feat = self.conv_first(x)
            body = self.conv_body(self.body(feat)) + feat
            out = F.leaky_relu(
                self.conv_up1(F.interpolate(body, scale_factor=2, mode="nearest")), 0.2, True)
            out = F.leaky_relu(
                self.conv_up2(F.interpolate(out, scale_factor=2, mode="nearest")), 0.2, True)
            return self.conv_last(F.leaky_relu(self.conv_hr(out), 0.2, True))

    return RRDBNet()


_net_cache = {}


def release_upscale() -> bool:
    """Drop cached Real-ESRGAN sessions so the VRAM is actually freed."""
    had = bool(_net_cache or _onnx_session)
    _net_cache.clear()
    _onnx_session.clear()
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return had


def _load_net(weight: Path, device: str):
    import torch
    key = (str(weight), device)
    net = _net_cache.get(key)
    if net is None:
        net = _rrdb_net()
        sd = torch.load(str(weight), map_location="cpu", weights_only=True)
        if isinstance(sd, dict):
            sd = sd.get("params_ema", sd.get("params", sd))
        if any(k.startswith("model.") for k in sd):
            sd = {k[len("model."):]: v for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)
        net.eval().to(device)
        for p in net.parameters():
            p.requires_grad_(False)
        _net_cache[key] = net
    return net


_onnx_session = {}


def _load_onnx_session(weight: Path):
    """Load or reuse an ONNX Runtime session for Real-ESRGAN."""
    import onnxruntime as ort
    from unified.onnx_env import pick_providers
    key = str(weight)
    sess = _onnx_session.get(key)
    if sess is None:
        provs = pick_providers("generic")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        if provs != ["CPUExecutionProvider"]:
            so.enable_mem_pattern = False
        sess = ort.InferenceSession(str(weight), sess_options=so, providers=provs)
        _onnx_session[key] = sess
    return sess


def _esrgan_upscale(img: Image.Image, weight: Path) -> Image.Image:
    """Torch-based tiled inference (x4)."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = _load_net(weight, device)
    w, h = img.size
    tile, overlap = 256, 16
    out = Image.new("RGB", (w * 4, h * 4))
    with torch.no_grad():
        for y in range(0, h, tile - overlap):
            for x in range(0, w, tile - overlap):
                x1, y1 = max(x - overlap, 0), max(y - overlap, 0)
                x2, y2 = min(x + tile + overlap, w), min(y + tile + overlap, h)
                crop = img.crop((x1, y1, x2, y2))
                t = torch.from_numpy(
                    __import__("numpy").asarray(crop).astype("float32") / 255.0
                ).permute(2, 0, 1).unsqueeze(0).to(device)
                sr = net(t).clamp(0, 1).squeeze(0).permute(1, 2, 0)
                sr = (sr.cpu().numpy() * 255.0 + 0.5).astype("uint8")
                sr_img = Image.fromarray(sr)
                # crop the overlap back to a clean tile
                ox1, oy1 = (x - x1) * 4, (y - y1) * 4
                cw = (min(x + tile - overlap, w) - x) * 4
                ch = (min(y + tile - overlap, h) - y) * 4
                out.paste(sr_img.crop((ox1, oy1, ox1 + cw, oy1 + ch)), (x * 4, y * 4))
                if device == "cuda":
                    torch.cuda.empty_cache()
    return out


def _esrgan_onnx_upscale(img: Image.Image, weight: Path) -> Image.Image:
    """ONNX-based tiled inference (x4)."""
    import numpy as np
    sess = _load_onnx_session(weight)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    w, h = img.size
    tile, overlap = 256, 16
    out = Image.new("RGB", (w * 4, h * 4))
    for y in range(0, h, tile - overlap):
        for x in range(0, w, tile - overlap):
            x1, y1 = max(x - overlap, 0), max(y - overlap, 0)
            x2, y2 = min(x + tile + overlap, w), min(y + tile + overlap, h)
            crop = img.crop((x1, y1, x2, y2))
            t = np.asarray(crop).astype(np.float32) / 255.0
            t = np.transpose(t, (2, 0, 1))[None, ...]  # NCHW
            sr = sess.run([output_name], {input_name: t})[0][0]
            sr = np.clip(sr, 0, 1)
            sr = np.transpose(sr, (1, 2, 0))
            sr = (sr * 255.0 + 0.5).astype(np.uint8)
            sr_img = Image.fromarray(sr)
            ox1, oy1 = (x - x1) * 4, (y - y1) * 4
            cw = (min(x + tile - overlap, w) - x) * 4
            ch = (min(y + tile - overlap, h) - y) * 4
            out.paste(sr_img.crop((ox1, oy1, ox1 + cw, oy1 + ch)), (x * 4, y * 4))
    return out


def upscale_image_b64(image_b64: str, scale: int = 2, model: str = "RealESRGAN_x4v3") -> dict:
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
    engine = "pillow"
    pref = (os.environ.get("PRISM_UPSCALE_ENGINE") or "auto").strip().lower()
    want_torch = model == "RealESRGAN_x4v3" and pref in ("auto", "torch")
    want_onnx = model == "RealESRGAN_x4v3" and pref in ("auto", "onnx")
    weight_torch = find_weight() if want_torch else None
    weight_onnx = find_onnx_weight() if want_onnx else None
    if weight_torch is not None:
        try:
            sr = _esrgan_upscale(img, weight_torch)
            engine = "realesrgan-torch"
            out = sr if scale == 4 else sr.resize((nw, nh), Image.LANCZOS)
        except Exception as e:
            logger.warning("realesrgan torch failed (%s), trying ONNX", str(e)[:140])
            if weight_onnx is not None:
                try:
                    sr = _esrgan_onnx_upscale(img, weight_onnx)
                    engine = "realesrgan-onnx"
                    out = sr if scale == 4 else sr.resize((nw, nh), Image.LANCZOS)
                except Exception as e2:
                    logger.warning("realesrgan onnx failed (%s), pillow fallback", str(e2)[:140])
                    out = img.resize((nw, nh), Image.LANCZOS)
            else:
                out = img.resize((nw, nh), Image.LANCZOS)
    elif weight_onnx is not None:
        try:
            sr = _esrgan_onnx_upscale(img, weight_onnx)
            engine = "realesrgan-onnx"
            out = sr if scale == 4 else sr.resize((nw, nh), Image.LANCZOS)
        except Exception as e:
            logger.warning("realesrgan onnx failed (%s), pillow fallback", str(e)[:140])
            out = img.resize((nw, nh), Image.LANCZOS)
    else:
        out = img.resize((nw, nh), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "elapsed": round(time.time() - t0, 2),
        "model": model,
        "scale": scale,
        "engine": engine,
    }
