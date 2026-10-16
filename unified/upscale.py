"""Upscale — Real-ESRGAN-x4v3 (RRDBNet) keď sú váha + torch, inak Pillow LANCZOS.

Váha: models/RealESRGAN_x4v3/RealESRGAN_x4plus.pth (BSD-3-Clause, 67 MB,
`download_models.py --preset upscale`). Dlaždicová inferencia (tile 256,
prekryv 16) drží VRAM dole — rovnaký vzor ako NCNN dlaždice na mobile.
DoS cap 8192px na stranu. scale 2/3: ESRGAN je x4 → výsledok sa downsampluje.
"""
import base64
import io
import time
from pathlib import Path

from PIL import Image

MAX_SIDE = 8192
WEIGHT_NAMES = ("RealESRGAN_x4plus.pth", "RealESRGAN_x4plus.pt",
                "RealESRGAN_x4plus.onnx")

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
            self.c1 = _conv(nf, gc)
            self.c2 = _conv(nf + gc, gc)
            self.c3 = _conv(nf + 2 * gc, gc)
            self.c4 = _conv(nf + 3 * gc, gc)
            self.c5 = _conv(nf + 4 * gc, nf)

        def forward(self, x):
            x1 = F.leaky_relu(self.c1(x), 0.2, True)
            x2 = F.leaky_relu(self.c2(torch.cat((x, x1), 1)), 0.2, True)
            x3 = F.leaky_relu(self.c3(torch.cat((x, x1, x2), 1)), 0.2, True)
            x4 = F.leaky_relu(self.c4(torch.cat((x, x1, x2, x3), 1)), 0.2, True)
            return self.c5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf=64, gc=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(nf, gc)
            self.rdb2 = ResidualDenseBlock(nf, gc)
            self.rdb3 = ResidualDenseBlock(nf, gc)

        def forward(self, x):
            return self.rdb1(x) + self.rdb2(x) + self.rdb3(x)

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = _conv(3, 64)
            self.body = nn.Sequential(*[RRDB(64) for _ in range(23)])
            self.conv_body = _conv(64, 64)
            self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                     _conv(64, 64))
            self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                                     _conv(64, 64))
            self.conv_hr = _conv(64, 64)
            self.conv_last = _conv(64, 3)

        def forward(self, x):
            feat = self.conv_first(x)
            body = self.conv_body(self.body(feat)) + feat
            out = self.up2(self.up1(body))
            return self.conv_last(self.conv_hr(out))

    return RRDBNet()


_net_cache = {}


def _load_net(weight: Path, device: str):
    import torch
    key = (str(weight), device)
    net = _net_cache.get(key)
    if net is None:
        net = _rrdb_net()
        sd = torch.load(str(weight), map_location="cpu", weights_only=True)
        if isinstance(sd, dict):
            sd = sd.get("params_ema", sd.get("params", sd))
        own = net.state_dict()
        sd = {k: v for k, v in sd.items() if k in own}
        own.update(sd)
        net.load_state_dict(own, strict=False)
        net.eval().to(device)
        for p in net.parameters():
            p.requires_grad_(False)
        _net_cache[key] = net
    return net


def _esrgan_upscale(img: Image.Image, weight: Path) -> Image.Image:
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
                # orez prekryvu späť na čistú dlaždicu
                ox1, oy1 = (x - x1) * 4, (y - y1) * 4
                cw = (min(x + tile - overlap, w) - x) * 4
                ch = (min(y + tile - overlap, h) - y) * 4
                out.paste(sr_img.crop((ox1, oy1, ox1 + cw, oy1 + ch)), (x * 4, y * 4))
                if device == "cuda":
                    torch.cuda.empty_cache()
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
    weight = find_weight() if model == "RealESRGAN_x4v3" else None
    if weight is not None:
        try:
            sr = _esrgan_upscale(img, weight)
            engine = "realesrgan"
            out = sr if scale == 4 else sr.resize((nw, nh), Image.LANCZOS)
        except Exception as e:
            import logging
            logging.getLogger("unified.upscale").warning(
                "realesrgan failed (%s), pillow fallback", str(e)[:140])
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
