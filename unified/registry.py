"""Registry — models.json (images) + Whisper list (audio) as one source."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

WHISPER_MODELS = ("tiny", "base", "small", "medium", "turbo", "large")
WHISPER_INFO = {
    "tiny": {"params": "39M", "vram": "~1 GB"},
    "base": {"params": "74M", "vram": "~1 GB"},
    "small": {"params": "244M", "vram": "~2 GB"},
    "medium": {"params": "769M", "vram": "~5 GB"},
    "turbo": {"params": "809M", "vram": "~6 GB"},
    "large": {"params": "1550M", "vram": "~10 GB"},
}


def load_image_registry():
    for name in ("models.json", "models.json.example"):
        p = HERE / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {"default_model": "DreamShaper_8", "models": {}, "port": 8101}


def build_registry():
    img = load_image_registry()
    return {
        "port": img.get("port", 8101),
        "default_image_model": img.get("default_model", "DreamShaper_8"),
        "default_audio_model": "small",
        "image_models": img.get("models", {}),
        "audio_models": list(WHISPER_MODELS),
        "audio_info": WHISPER_INFO,
        "negative_prompt": img.get("negative_prompt", ""),
    }


def write_registry_json(path=None):
    p = Path(path) if path else HERE / "registry.json"
    p.write_text(json.dumps(build_registry(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p
