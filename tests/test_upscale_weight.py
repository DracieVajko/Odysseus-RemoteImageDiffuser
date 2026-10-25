"""Upscale — weight discovery is a pure function (testable without torch)."""
from unified.upscale import find_weight, upscale_image_b64
import base64
import io
from PIL import Image


def _red_dot_b64(size=16):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_find_weight_none_when_absent(tmp_path):
    assert find_weight([tmp_path]) is None


def test_find_weight_finds_pth(tmp_path):
    w = tmp_path / "RealESRGAN_x4plus.pth"
    with open(w, "wb") as f:
        f.truncate(11 * 1024 * 1024)  # sparse, only to pass the size guard
    assert find_weight([tmp_path]) == w


def test_upscale_reports_engine(monkeypatch):
    monkeypatch.setenv("PRISM_UPSCALE_ENGINE", "pillow")
    out = upscale_image_b64(_red_dot_b64(16), 2)
    assert out["engine"] == "pillow", out.get("engine")
    raw = base64.b64decode(out["image"])
    assert Image.open(io.BytesIO(raw)).size == (32, 32)
