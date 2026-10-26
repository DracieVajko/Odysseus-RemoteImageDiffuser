"""Engine init + real inpaint delegation.

diffusion_server.py keeps globals (_pipe, _args, registry) and cannot infer
without a CLI init. init_diffusion_engine() initializes it in lazy mode
(registry from models.json, no weights) by setting a minimal _args
Namespace — called once at unified_server startup, never on import.
Without torch/diffusers it returns False and unified_server uses the placeholder/stub path.
"""
import argparse
import base64
import logging

logger = logging.getLogger("unified.engine")
_inited = False


def _have_torch():
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except Exception:
        return False


def init_diffusion_engine() -> bool:
    global _inited
    if _inited:
        return True
    if not _have_torch():
        return False
    try:
        import diffusion_server as ds
        # diffusion_server owns the argparse defaults for embedded use, so the
        # registry is initialised through one code path (no second Namespace
        # that can drift out of sync with the CLI).
        ds.ensure_registry()
        _inited = True
        return True
    except Exception as e:
        logger.warning(f"diffusion engine init failed: {str(e)[:160]}")
        return False


def _is_engine_ready() -> bool:
    if not _inited or not _have_torch():
        return False
    try:
        import diffusion_server as ds
        return getattr(ds, "_pipe", None) is not None or bool(getattr(ds, "_REGISTRY", {}))
    except Exception:
        return False


def try_real_inpaint(image_b64, mask_b64, prompt, model_id,
                     strength=0.75, steps=0, force_engine=False):
    """Return the resulting image b64 from the real pipeline, or None (echo fallback).

    ValueError on bad base64 input — always, even without the engine (validation first)."""
    try:
        base64.b64decode(image_b64)
        base64.b64decode(mask_b64)
    except Exception as e:
        raise ValueError(f"bad base64 image/mask: {e}")
    if not _is_engine_ready():
        if force_engine:
            raise RuntimeError("diffusion engine not initialized")
        return None
    import diffusion_server as ds
    import uuid
    ds.ensure_model_loaded(model_id)
    req = ds.InpaintRequest(
        image=image_b64, mask=mask_b64, prompt=prompt or "",
        strength=float(strength or 0.75), steps=int(steps or 0),
        model=model_id or "", request_id=f"unified-{uuid.uuid4().hex[:8]}",
    )
    out = ds._inpaint_locked(req)
    if isinstance(out, dict) and out.get("image"):
        return out["image"]
    return None
