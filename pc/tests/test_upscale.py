"""Upscale — Real-ESRGAN-x4v3 default, Pillow fallback s tile hookom."""
import base64
import io
import pytest
from PIL import Image
from unified.upscale import upscale_image_b64


def _red_dot_b64(size=16):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_upscale_2x_grows_canvas():
    out = upscale_image_b64(_red_dot_b64(16), 2)
    raw = base64.b64decode(out["image"])
    img = Image.open(io.BytesIO(raw))
    assert img.size == (32, 32)
    assert out["elapsed"] >= 0


def test_upscale_rejects_absurd():
    with pytest.raises(ValueError):
        upscale_image_b64(_red_dot_b64(8), 64)
