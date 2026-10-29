"""Gemma4 text post-processing for Prism (PC).

Provides text cleanup, translation, and rewriting using Google Gemma4 E2B.
Used to improve transcription quality: fix errors, add punctuation,
translate, or summarize transcribed text.

Model: google/gemma-4-E2B-it (~2.5B params, Apache 2.0)
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("gemma_postprocess")

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_DIR = None  # set by init

_processor = None
_model = None
_device = None


def _get_model_dir() -> Path:
    global MODEL_DIR
    if MODEL_DIR is not None:
        return MODEL_DIR
    # Default: pc/models/gemma-4-E2B-it
    base = Path(__file__).resolve().parent.parent / "models"
    MODEL_DIR = base / "gemma-4-E2B-it"
    return MODEL_DIR


def is_installed() -> bool:
    d = _get_model_dir()
    return (d / "model.safetensors").exists() or (d / "config.json").exists()


def init(device: str = "auto") -> bool:
    """Load Gemma4 model. Returns True on success."""
    global _processor, _model, _device
    if _model is not None:
        return True
    if not is_installed():
        logger.warning("Gemma4 not installed at %s", _get_model_dir())
        return False
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        d = _get_model_dir()
        _device = "cuda" if (device == "auto" and torch.cuda.is_available()) else (device if device != "auto" else "cpu")
        dtype = torch.bfloat16 if _device == "cuda" else torch.float32
        _processor = AutoProcessor.from_pretrained(str(d))
        _model = AutoModelForImageTextToText.from_pretrained(str(d), dtype=dtype, device_map=_device)
        logger.info("Gemma4 loaded on %s", _device)
        return True
    except Exception as e:
        logger.error("Gemma4 init failed: %s", e)
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


def process(text: str, instruction: str = "cleanup", target_lang: str = "") -> str:
    """Post-process text with Gemma4.

    instruction: 'cleanup' | 'translate' | 'summarize' | 'rewrite'
    target_lang: target language for translation (e.g. 'en', 'sk')
    """
    if _model is None:
        if not init():
            return text
    try:
        import torch
        prompts = {
            "cleanup": (
                "Fix spelling, grammar, and punctuation in the following "
                "transcription. Keep the original language. Return only the "
                "corrected text, nothing else.\n\n{text}"
            ),
            "translate": (
                "Translate the following text to {lang}. Return only the "
                "translation, nothing else.\n\n{text}"
            ),
            "summarize": (
                "Summarize the following text in a few sentences. "
                "Return only the summary.\n\n{text}"
            ),
            "rewrite": (
                "Rewrite the following text to be clearer and more readable, "
                "while preserving the meaning. Return only the rewritten text.\n\n{text}"
            ),
        }
        template = prompts.get(instruction, prompts["cleanup"])
        lang_name = {"sk": "Slovak", "cs": "Czech", "en": "English", "de": "German",
                     "fr": "French", "es": "Spanish", "it": "Italian", "pl": "Polish",
                     "ru": "Russian", "uk": "Ukrainian", "hu": "Hungarian"}.get(target_lang, target_lang)
        prompt = template.format(text=text, lang=lang_name)
        messages = [{"role": "user", "content": prompt}]
        inputs = _processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = _model.generate(**inputs, max_new_tokens=2048, do_sample=False)
        result = _processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return result.strip()
    except Exception as e:
        logger.error("Gemma4 process failed: %s", e)
        return text
