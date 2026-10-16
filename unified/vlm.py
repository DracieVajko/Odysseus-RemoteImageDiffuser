"""VLM caption (V2) — SmolVLM-500M-Instruct cez transformers, lazy.

Váha: models/SmolVLM_500M/ (`download_models.py --preset vlm`, ~1 GB, Apache-2.0).
Bez váhy: EngineMissing → endpoint vráti 503 will-not-load (nie 500).
Prepnutie motorov (SD↔VLM) rieši unified_server cez release_vlm() /
diffusion_server.unload_current_model() — vždy max 1 ťažký model.
"""
import base64
import io
import logging
import time
from pathlib import Path

from PIL import Image

logger = logging.getLogger("unified.vlm")

_HERE = Path(__file__).resolve().parent.parent
_DEFAULT_DIRS = (_HERE / "models" / "SmolVLM_500M",)
MODEL_ID = "SmolVLM_500M"
MAX_SIDE = 1024


class EngineMissing(Exception):
    pass


def find_vlm_dir(dirs=None):
    for d in (dirs if dirs is not None else _DEFAULT_DIRS):
        d = Path(d)
        if (d / "model.safetensors").is_file() or (d / "pytorch_model.bin").is_file():
            return d
    return None


_cache = {}


def _have_torch_stack() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


def _load(model_dir: Path):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    key = str(model_dir)
    if key not in _cache:
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        dtype = torch.float16 if cuda else torch.float32
        logger.info(f"loading VLM {MODEL_ID} from {model_dir} ({device})")
        proc = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=False)
        model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), dtype=dtype, trust_remote_code=False).to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _cache[key] = (proc, model, device)
    return _cache[key]


def release_vlm() -> bool:
    had = bool(_cache)
    _cache.clear()
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


def caption_image_b64(image_b64: str, prompt: str = "Describe the image in detail.",
                      max_tokens: int = 256, search_dirs=None) -> dict:
    try:
        raw = base64.b64decode(image_b64)
    except Exception as e:
        raise ValueError(f"bad base64 image: {e}")
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"bad image bytes: {e}")
    model_dir = find_vlm_dir(search_dirs)
    if model_dir is None:
        raise EngineMissing(f"VLM weights missing ({MODEL_ID}); run download_models.py --preset vlm")
    if not _have_torch_stack():
        raise EngineMissing("torch/transformers not installed in this python env")
    if img.width > MAX_SIDE or img.height > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    t0 = time.time()
    proc, model, device = _load(model_dir)
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt or "Describe the image."}]}]
    # Šablóna vyrobí text s <image> tokenom; obraz sa pasuje zvlášť (nie return_dict).
    text_in = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=text_in, images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max(16, min(512, max_tokens)))
    gen = out[0][inputs["input_ids"].shape[1]:]
    text = proc.batch_decode([gen], skip_special_tokens=True)[0].strip()
    return {"text": text, "elapsed": round(time.time() - t0, 2), "engine": "smolvlm"}
