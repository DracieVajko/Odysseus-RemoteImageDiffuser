"""Engine init + real inpaint delegation.

diffusion_server.py drží globály (_pipe, _args, registry) a bez CLI init
nevie inferovať. init_diffusion_engine() ho inicializuje v lazy režime
(registry z models.json, žiadne váhy) nastavením minimálneho _args
Namespace — volá sa raz pri štarte unified_server, nikdy pri importe.
Bez torch/diffusers vráti False a unified_server ide placeholder/stub cestou.
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
        if getattr(ds, "_args", None) is None:
            ds._args = argparse.Namespace(
                model="", lazy=True, models_json="", default_model="",
                steps=25, guidance_scale=7.0, width=768, height=768,
                dtype="float16", pipeline="StableDiffusionPipeline",
                attention_slicing=True, vae_slicing=True, cpu_offload=False,
                harmonize_gpu=None, negative_prompt="", single_file_config="",
                disable_safety_checker=True, performance="auto",
            )
        ds._init_lazy()
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
    """Vráti výsledný image b64 z reálnej pipeline, alebo None (fallback echo).

    ValueError pri zlom base64 vstupe — vždy, aj bez enginu (validácia prvá).
    """
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
