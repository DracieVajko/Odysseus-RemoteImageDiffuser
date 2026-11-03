"""NVIDIA Nemotron 3.5 ASR (0.6B) as an alternative PC transcription engine.

Model: nvidia/nemotron-3.5-asr-streaming-0.6b (FastConformer-RNNT, 40 languages
with native punctuation and capitalisation, ~2.5 GB in bfloat16).

Enabled with PRISM_ASR_ENGINE=nemotron, or per-request with
`model=nemotron-asr-0.6b` on /v1/audio/transcriptions.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("nemotron_asr")

MODEL_ID = "nemotron-asr-0.6b"
MAX_SECONDS_PER_CALL = 40.0  # the model is trained on <=40 s windows

# Directory names must match download_models.AUDIO_TASKS targets.
MODEL_DIRS = {
    "nemotron-asr-0.6b": "nemotron-asr-0.6b",
    "nemotron-transcribe-30b-a4b-nvfp4": "nemotron-transcribe-30b-a4b-nvfp4",
    "nemotron-transcribe-30b-a4b-fp8": "nemotron-transcribe-30b-a4b-fp8",
}

# Rough resident-memory requirement per variant, used to fail with a clear
# message instead of letting the loader OOM the machine.
MODEL_MIN_RAM_GB = {
    "nemotron-asr-0.6b": 4.0,
    "nemotron-transcribe-30b-a4b-nvfp4": 24.0,
    "nemotron-transcribe-30b-a4b-fp8": 40.0,
}

_active_id = MODEL_ID


def _model_dir(model_id: str = None) -> Path:
    mid = model_id or _active_id
    sub = MODEL_DIRS.get(mid, mid)
    return Path(__file__).resolve().parent.parent / "models" / sub


def is_installed(model_id: str = None) -> bool:
    d = _model_dir(model_id)
    has_weights = (d / "model.safetensors").exists() or (d / "model.safetensors.index.json").exists()
    return has_weights and (d / "config.json").exists()


def _total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 0.0


def enabled() -> bool:
    return os.environ.get("PRISM_ASR_ENGINE", "").strip().lower() == "nemotron"


_processor = None
_model = None
_device = None


def _auto_device():
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            if free > 3 * 1024 ** 3:
                return "cuda"
    except Exception:
        pass
    return "cpu"


def init(model_id: str = None) -> bool:
    """Load a Nemotron variant. Returns False and logs the reason on failure."""
    global _processor, _model, _device, _active_id
    mid = model_id or _active_id
    if _model is not None and mid == _active_id:
        return True
    if _model is not None and mid != _active_id:
        release()
    _active_id = mid
    if not is_installed(mid):
        logger.warning("Nemotron ASR not installed at %s", _model_dir(mid))
        return False
    needed = MODEL_MIN_RAM_GB.get(mid, 4.0)
    have = _total_ram_gb()
    if have and have + 0.5 < needed:
        logger.error("Nemotron %s needs ~%.0f GB RAM, this machine has %.1f GB",
                     mid, needed, have)
        return False
    try:
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor

        d = str(_model_dir(mid))
        _device = (os.environ.get("PRISM_NEMOTRON_DEVICE") or _auto_device()).strip().lower()
        dtype = torch.bfloat16 if _device == "cuda" else torch.float32
        _processor = AutoProcessor.from_pretrained(d)
        _model = AutoModelForRNNT.from_pretrained(d, dtype=dtype, device_map=_device)
        logger.info("Nemotron ASR %s loaded on %s", mid, _device)
        return True
    except Exception as e:
        logger.error("Nemotron ASR init failed: %s", e)
        _processor = None
        _model = None
        return False


def release():
    global _processor, _model, _device
    _processor = None
    _model = None
    _device = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _normalise_language(code: Optional[str]) -> str:
    """Map a Whisper-style code to one the Nemotron prompt dictionary knows."""
    if not code:
        return "auto"
    c = code.strip().lower().replace("_", "-")
    if c in ("auto", "detect", ""):
        return "auto"
    # exact match first, then the base language
    known = {
        "af", "am", "ar", "ay", "az", "bg", "bn", "cs", "da", "de", "el", "en",
        "es", "et", "fa", "fi", "fr", "gn", "gu", "ha", "haw", "he", "hi", "hr",
        "hu", "hy", "id", "ig", "it", "ja", "ka", "km", "kn", "ko", "ku", "ky",
        "ln", "lt", "lv", "mi", "ml", "mr", "ms", "mt", "nah", "nb", "ne", "nl",
        "nn", "no", "ny", "or", "pl", "pt", "qu", "ro", "ru", "rw", "si", "sk",
        "sl", "sm", "so", "sv", "sw", "ta", "te", "tg", "th", "to", "tr", "uk",
        "ur", "uz", "vi", "yo", "zh", "zu",
    }
    base = c.split("-")[0]
    if base in known:
        return base
    return "auto"


def transcribe(audio_path: str, language: Optional[str] = None,
               model_id: str = None) -> tuple[str, str]:
    """Transcribe one file. Returns (text, language_used)."""
    if model_id and model_id != _active_id:
        if not init(model_id):
            raise RuntimeError(f"nemotron model {model_id} not available")
    elif _model is None and not init():
        raise RuntimeError("nemotron asr not available")

    import torch
    from transformers.audio_utils import load_audio

    lang = _normalise_language(language)
    audio = load_audio(str(audio_path), sampling_rate=16000)
    if audio is None or len(audio) == 0:
        raise RuntimeError("empty audio")

    # The checkpoint can be bf16 on CUDA; the feature extractor always emits
    # float32, so cast the floating point inputs to the model's own dtype.
    model_dtype = None
    try:
        model_dtype = next(_model.parameters()).dtype
    except Exception:
        pass

    chunk = int(MAX_SECONDS_PER_CALL * 16000)
    parts: list[str] = []
    for start in range(0, len(audio), chunk):
        piece = audio[start:start + chunk]
        inputs = _processor(piece, sampling_rate=16000, language=lang, return_tensors="pt")
        for k, v in list(inputs.items()):
            if not hasattr(v, "to"):
                continue
            v = v.to(_model.device)
            if model_dtype is not None and v.is_floating_point() and v.dtype != model_dtype:
                v = v.to(model_dtype)
            inputs[k] = v
        with torch.no_grad():
            out = _model.generate(**inputs, max_new_tokens=1024)
        text = _processor.decode(out[0], skip_special_tokens=True)
        if isinstance(text, (list, tuple)):
            text = " ".join(text)
        text = text.strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip(), lang
