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


def _model_bytes() -> int:
    d = _get_model_dir()
    total = 0
    for p in d.glob("*.safetensors"):
        try:
            total += p.stat().st_size
        except OSError:
            continue
    return total


def _cuda_fits() -> bool:
    """True when the weights fit in the GPU's free VRAM with headroom.

    A partial device_map split across CUDA and CPU is what produced
    'CUDA driver error: device not ready' with the 9.8 GB E2B checkpoint on a
    6 GB card, so we only pick CUDA when the whole model fits.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        free, _total = torch.cuda.mem_get_info()
        needed = _model_bytes()
        # 1.15x headroom for the KV cache and activations
        return needed > 0 and needed * 1.15 < free
    except Exception:
        return False


def _load(device: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    d = _get_model_dir()
    if device == "cuda":
        import torch
        return (
            AutoProcessor.from_pretrained(str(d)),
            AutoModelForImageTextToText.from_pretrained(
                str(d), dtype=torch.bfloat16, device_map="cuda"),
        )
    # CPU: bfloat16 keeps the resident footprint at ~10 GB (float32 would need
    # ~20 GB and push a 32 GB machine into swap). PyTorch supports bf16 CPU
    # kernels on every modern x86 build.
    import torch
    return (
        AutoProcessor.from_pretrained(str(d)),
        AutoModelForImageTextToText.from_pretrained(
            str(d), dtype=torch.bfloat16, device_map="cpu"),
    )


def init(device: str = "auto") -> bool:
    """Load Gemma4. Falls back to CPU when the weights do not fit in VRAM."""
    global _processor, _model, _device
    if _model is not None:
        return True
    if not is_installed():
        logger.warning("Gemma4 not installed at %s", _get_model_dir())
        return False

    wanted = (device or "auto").lower()
    if wanted == "auto":
        wanted = "cuda" if _cuda_fits() else "cpu"
        logger.info("Gemma4 device auto -> %s (weights %.1f GB)",
                    wanted, _model_bytes() / 1e9)

    last_error = None
    for candidate in ([wanted, "cpu"] if wanted == "cuda" else [wanted]):
        try:
            _processor, _model = _load(candidate)
            _device = candidate
            logger.info("Gemma4 loaded on %s", candidate)
            return True
        except Exception as e:
            last_error = e
            logger.warning("Gemma4 load on %s failed: %s", candidate, str(e)[:200])
            _processor = None
            _model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    logger.error("Gemma4 init failed: %s", last_error)
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


def _language_name(code: str) -> str:
    return {
        "sk": "Slovak", "cs": "Czech", "en": "English", "de": "German",
        "fr": "French", "es": "Spanish", "it": "Italian", "pl": "Polish",
        "ru": "Russian", "uk": "Ukrainian", "hu": "Hungarian",
    }.get(code, code)


def _plausible(output: str, source: str) -> bool:
    """Reject degenerate generations that would make the endpoint worse.

    Greedy decoding on a bf16 CPU copy occasionally loops or emits token soup
    (replacement characters). Falling back to the source text is always safer
    than returning garbage to the caller.
    """
    if not output.strip():
        return False
    if "\ufffd" in output:
        return False
    # A cleanup/rewrite must stay in the same order of magnitude as the input.
    if len(output) > max(64, len(source) * 3):
        return False
    return True


def process(text: str, instruction: str = "cleanup", target_lang: str = "") -> str:
    """Post-process text with Gemma4.

    instruction: 'cleanup' | 'translate' | 'summarize' | 'rewrite'
    target_lang: target language for translation (e.g. 'en', 'sk')
    """
    if not text.strip():
        return text
    if _model is None:
        if not init():
            return text
    try:
        import torch
        lang_name = _language_name(target_lang)
        prompts = {
            "cleanup": (
                "You are a transcription editor. Correct the spelling, grammar "
                "and punctuation of the {lang} text below. Keep it in {lang}. "
                "Do not add commentary, do not translate, do not repeat the "
                "instructions. Reply with the corrected text only.\n\n"
                "Text:\n{text}\n\nCorrected text:"
            ),
            "translate": (
                "Translate the text below into {lang}. Reply with the "
                "translation only, without commentary.\n\n"
                "Text:\n{text}\n\nTranslation:"
            ),
            "summarize": (
                "Summarize the text below in a few sentences. Reply with the "
                "summary only.\n\nText:\n{text}\n\nSummary:"
            ),
            "rewrite": (
                "Rewrite the text below so it is clearer and easier to read, "
                "keeping the meaning and the original language. Reply with the "
                "rewritten text only.\n\nText:\n{text}\n\nRewritten text:"
            ),
        }
        template = prompts.get(instruction, prompts["cleanup"])
        prompt = template.format(text=text, lang=lang_name)
        messages = [{"role": "user", "content": prompt}]
        inputs = _processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        # Bound the generation: a cleanup should never need more than roughly
        # 1.5x the input, a summary needs less. Without this the model can
        # ramble for hundreds of tokens.
        prompt_tokens = int(inputs["input_ids"].shape[1])
        if instruction == "summarize":
            budget = 200
        else:
            budget = max(48, min(1024, prompt_tokens))
        budget = min(budget, 1024)

        with torch.no_grad():
            out = _model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=getattr(_processor.tokenizer, "pad_token_id", None)
                or getattr(_processor.tokenizer, "eos_token_id", None),
            )
        result = _processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if not _plausible(result, text):
            logger.warning("Gemma4 %s output rejected (len=%d), returning source",
                           instruction, len(result))
            return text
        return result
    except Exception as e:
        logger.error("Gemma4 process failed: %s", e)
        return text
