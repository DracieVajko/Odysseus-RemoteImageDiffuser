"""Convert RealESRGAN_x4plus.pth to ONNX (torch is only needed for conversion).

Usage:
    python convert_esrgan_onnx.py

Output: models/RealESRGAN_x4v3/RealESRGAN_x4plus.onnx
The ONNX file is used by unified/upscale.py without torch.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from unified.upscale import _load_net, find_weight


def main() -> int:
    weight = find_weight()
    if weight is None:
        print("RealESRGAN .pth not found; run download_models.py --preset upscale")
        return 1
    if weight.suffix.lower() not in (".pth", ".pt"):
        print(f"unexpected weight: {weight}")
        return 1
    out = weight.with_suffix(".onnx")

    net = _load_net(weight, "cpu")

    dummy = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        torch.onnx.export(
            net, dummy, str(out),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch", 2: "h", 3: "w"},
                          "output": {0: "batch", 2: "h", 3: "w"}},
            opset_version=17,
            do_constant_folding=True,
        )
    print("wrote", out, out.stat().st_size, "bytes")
    data = out.with_suffix(".onnx.data")
    if data.is_file():
        # Inline the external weights so the model is a single portable file.
        import onnx
        model = onnx.load(str(out))
        onnx.save_model(model, str(out), save_as_external_data=False)
        data.unlink(missing_ok=True)
        print("inlined external weights ->", out.stat().st_size, "bytes")

    # Validate: ONNX vs torch on a real image crop (random noise explodes).
    import numpy as np
    from PIL import Image
    import onnxruntime as ort
    sample = Path(__file__).resolve().parent / "testdata" / "images" / "test_512.png"
    if sample.is_file():
        crop = Image.open(sample).convert("RGB").crop((0, 0, 64, 64))
        x = (np.asarray(crop).astype("float32") / 255.0).transpose(2, 0, 1)[None, ...]
    else:
        x = np.full((1, 3, 64, 64), 0.5, dtype=np.float32)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()
    got = sess.run(None, {"input": x})[0]
    diff = float(np.abs(ref - got).max())
    print(f"output range: {float(ref.min()):.3f}..{float(ref.max()):.3f}")
    print(f"max abs diff vs torch: {diff:.6f}")
    if diff > 1e-3:
        print("WARNING: ONNX output differs from torch")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
