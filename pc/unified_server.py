#!/usr/bin/env python3
"""Unified Standalone Server — images + audio in one process, one contract.


Combines DiffusionServer (:8101) + WhisperApi (:4444) without duplicating
heavy logic: delegates to existing lazy modules (behind the ModelManager
gate), so idle VRAM/RAM stays free. At most one heavy model resident.
Routes (frozen for Odysseus):
GET /health, GET /v1/models, POST /v1/images/generations|edits,
POST /v1/images/inpaint|harmonize|upscale, POST /v1/audio/transcriptions|translations,
GET /v1/images/progress/{id}, GET /webui, GET / -> /webui
"""
import asyncio
import base64
import io
import tempfile
import threading
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse)
from pydantic import BaseModel
from typing import Optional, Tuple, List

from unified.audio_proxy import transcribe_via_service
from unified.audio_splitter import transcribe_with_splitting, get_audio_duration
from unified.ffmpeg_tools import (
    VIDEO_PRESETS, AUDIO_PRESETS, IMAGE_PRESETS, RESIZE_PRESETS,
    convert_video, convert_audio, convert_image, extract_audio, trim_media, compress_video,
    get_media_info, get_presets,
)
from unified.engine import init_diffusion_engine, try_real_inpaint
from unified.model_manager import ModelManager, ModelLoadingError
from unified.registry import build_registry
from unified.upscale import upscale_image_b64
from unified.vlm import MODEL_ID as VLM_ID, caption_image_b64, EngineMissing, release_vlm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified_server")

HERE = Path(__file__).resolve().parent
WHISPER_SERVICE_URL = os.getenv("UNIFIED_WHISPER_URL", "http://127.0.0.1:4444")
_manager = ModelManager()
_START_TS = time.time()


IDLE_UNLOAD_SECONDS = int(os.getenv("UNIFIED_IDLE_UNLOAD_SECONDS", "900"))


def _release_engines(reason: str = ""):
    """Release real weights on switch/idle (the manager keeps bookkeeping only)."""
    try:
        import diffusion_server as ds
        if getattr(ds, "_pipe", None) is not None:
            ds.unload_current_model()
    except Exception:
        pass
    try:
        from unified.lcm_onnx_numpy import release_pipeline
        release_pipeline()
    except Exception:
        pass
    try:
        from unified.upscale import release_upscale
        release_upscale()
    except Exception:
        pass
    try:
        release_whisper_cache()
    except Exception:
        pass
    try:
        release_vlm()
    except Exception:
        pass


def should_idle_unload(mm, limit: int) -> bool:
    """Idle unload only when a model is resident, nothing runs, and the limit is exceeded."""
    if mm.status()["loaded_kind"] is None:
        return False
    if mm.in_flight() > 0:
        return False
    return mm.idle_seconds() >= limit


async def _idle_unloader():
    while True:
        await asyncio.sleep(30)
        try:
            if not should_idle_unload(_manager, IDLE_UNLOAD_SECONDS):
                continue
            if True:
                logger.info("idle unload after %.0fs (limit %ss)",
                            _manager.idle_seconds(), IDLE_UNLOAD_SECONDS)
                _manager.unload("idle")
        except Exception as e:
            logger.warning("idle unloader: %s", e)


@asynccontextmanager
async def lifespan(application):
    # Best effort: registry without weights; real inference only with torch + weights.
    try:
        if os.getenv("UNIFIED_NO_ENGINE", "") != "1":
            init_diffusion_engine()
    except Exception as e:
        logger.warning(f"engine init skipped: {e}")
    _manager.on_unload = _release_engines
    task = asyncio.create_task(_idle_unloader())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Unified Standalone Server", lifespan=lifespan)

_TRACKED_PREFIXES = ("/v1/images/", "/v1/audio/", "/v1/chat/")


@app.middleware("http")
async def _track_inflight(request, call_next):
    """Counts in-flight inference requests so the idle unloader cannot
    evict a model during a long generation (cold load ~3 min > idle limit)."""
    path = request.url.path
    track = request.method == "POST" and path.startswith(_TRACKED_PREFIXES)
    if track:
        _manager.begin()
    try:
        return await call_next(request)
    finally:
        if track:
            _manager.end()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- helpers ----------

def _placeholder_png_b64(size=(64, 64), color=(200, 30, 30)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        from unified.onnx_env import available_providers
        provs = available_providers()
        if "CUDAExecutionProvider" in provs:
            return "cuda"
        if "DmlExecutionProvider" in provs:
            return "directml"
    except Exception:
        pass
    return "cpu"


def _onnx_provider_label():
    try:
        from unified.onnx_env import available_providers, provider_label, pick_providers
        return provider_label(pick_providers("qdq"))
    except Exception:
        return "unknown"


def _image_registry():
    try:
        return build_registry()
    except Exception:
        return {"image_models": {}, "default_image_model": "DreamShaper_8",
                "audio_models": ["tiny", "base", "small", "medium", "turbo", "large"],
                "default_audio_model": "small"}


def _engines_disabled() -> bool:
    v = os.environ.get("UNIFIED_NO_ENGINE", "")
    return v not in ("", "0", "false", "False", "no")


def _raise_loading(e: ModelLoadingError):
    return JSONResponse(
        status_code=503,
        content={"error": "loading model", "loading_model": e.loading_model,
                 "retry_after": e.retry_after},
    )


# ---------- models ----------

class GenRequest(BaseModel):
    model: Optional[str] = ""
    prompt: str = ""
    negative_prompt: str = ""
    guidance: float = 1.0
    n: int = 1
    size: str = "768x768"
    quality: str = "medium"
    request_id: Optional[str] = ""


class InpaintRequest(BaseModel):
    image: str = ""
    mask: Optional[str] = ""
    prompt: str = ""
    negative_prompt: str = ""
    guidance: float = 1.0
    model: Optional[str] = ""
    strength: float = 0.75
    steps: int = 0
    mode: str = "add"          # add | remove | color
    color: str = "#0066ff"     # target color for mode=color
    width: Optional[int] = None
    height: Optional[int] = None
    request_id: Optional[str] = ""


class OutpaintRequest(BaseModel):
    """Expand the canvas and generate the new border with the inpaint engine."""
    image: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    model: Optional[str] = ""
    # Pixels per side; `margin` (fraction of the shorter side) is used for the
    # sides that are left at 0.
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0
    margin: float = 0.0
    guidance: float = 1.0
    strength: float = 0.99
    steps: int = 6
    request_id: Optional[str] = ""


class HarmonizeRequest(BaseModel):
    image: str = ""
    prompt: str = ""
    model: Optional[str] = ""
    color_match: bool = True
    seam_fix: bool = True
    body_mask: Optional[str] = ""
    seam_mask: Optional[str] = ""
    mask: Optional[str] = ""
    strength: float = 0.5
    max_side: int = 1024
    steps: int = 0
    request_id: Optional[str] = ""


class UpscaleRequest(BaseModel):
    image: str = ""
    scale: int = 2
    model: Optional[str] = "RealESRGAN_x4v3"
    request_id: Optional[str] = ""


class CaptionRequest(BaseModel):
    image: str = ""
    prompt: Optional[str] = "Describe the image in detail."
    max_tokens: int = 256
    model: Optional[str] = ""
    request_id: Optional[str] = ""


def _release_vlm_best_effort():
    try:
        release_vlm()
    except Exception:
        pass


def _release_sd_best_effort():
    try:
        import diffusion_server as ds
        if getattr(ds, "_pipe", None) is not None:
            ds.unload_current_model()
    except Exception:
        pass




from contextlib import contextmanager


@contextmanager
def _in_flight():
    _manager.begin()
    try:
        yield
    finally:
        _manager.end()


# ---------- system ----------

@app.get("/")
def root():
    return RedirectResponse(url="/webui")


@app.get("/health")
def health():
    reg = _image_registry()
    st = _manager.status()
    actual_model, actual_kind = "", None
    try:
        import diffusion_server as ds
        if getattr(ds, "_pipe", None) is not None and getattr(ds, "_model_id", ""):
            actual_model, actual_kind = ds._model_id, "image"
    except Exception:
        pass
    if not actual_model and st["loaded_kind"] == "vlm":
        actual_model, actual_kind = VLM_ID, "vlm"
    if not actual_model and st["loaded_kind"] == "audio":
        actual_model, actual_kind = st["loaded_id"], "audio"
    # faster-whisper weights live in _WHISPER_CACHE, not in the manager, so
    # report what is actually resident instead of the bookkeeping kind.
    whisper_name = ""
    try:
        if _WHISPER_CACHE:
            src = str(next(iter(_WHISPER_CACHE))[0])
            whisper_name = Path(src).name.replace("faster-whisper-", "") or src
    except Exception:
        whisper_name = ""
    return {
        "status": "ok",
        "model": reg.get("default_image_model", ""),
        "current_model": actual_model or st["loaded_id"],
        "device": _device(),
        "onnx_provider": _onnx_provider_label(),
        "available_models": sorted(list(reg.get("image_models", {}).keys())),
        "loading": st["loading"],
        "loading_model": st["loading_model"],
        "busy": st["busy"],
        "queue": st["queue"],
        "in_flight": st.get("in_flight"),
        "idle_seconds": st.get("idle_seconds"),
        "idle_unload_seconds": IDLE_UNLOAD_SECONDS,
        "uptime_s": round(time.time() - _START_TS, 1),
        # audio sekcia (WhisperApi kompat)
        "audio_model": whisper_name or reg.get("default_audio_model", "small"),
        "audio_loaded": bool(whisper_name) or st["loaded_kind"] == "audio",
        "audio_whisper_models_resident": len(_WHISPER_CACHE),
        "version": "unified-0.1.0",
    }


@app.get("/v1/models")
def list_models():
    reg = _image_registry()
    st = _manager.status()
    models = reg.get("image_models", {})
    data = []
    # The installable catalog is the source of truth for `installed` and kind;
    # build_registry's `missing` flag is not always populated.
    try:
        from unified import models_manager as _mm
    except Exception:
        _mm = None
    try:
        cat = {m["id"]: m for m in _mm.catalog()} if _mm else {}
    except Exception:
        cat = {}
    for k in sorted(models.keys()):
        entry = models.get(k, {})
        c = cat.get(k, {})
        installed = bool(c.get("installed", not bool(entry.get("missing"))))
        data.append({
            "id": k, "object": "model", "owned_by": "unified",
            "kind": c.get("kind", entry.get("kind", "image")),
            "label": c.get("label", entry.get("label", k)),
            "engine": c.get("engine", entry.get("engine", "")),
            "installed": installed,
            "available": installed,
            "size_mb": c.get("size_mb", 0),
            "install_state": c.get("install_state", "idle"),
            "install_progress": c.get("install_progress", 0),
            "downloadable": bool(c),
            "has_source": bool(c),
        })
    # Audio: faster-whisper names. The installable manager is the single source
    # of truth (exact model.bin size + valid config.json), so an incomplete
    # download never looks installed and the UI can offer a real button.
    # NOTE: base the list on reg["audio_models"] (the supported sizes), not on
    # the catalog, so an entry is never added twice.
    for a in reg.get("audio_models", []):
        entry = {}
        if _mm is not None:
            try:
                entry = _mm.whisper_entry(a)
            except Exception:
                entry = {}
        data.append({
            "id": a, "object": "model", "owned_by": "unified",
            "kind": "audio",
            "label": entry.get("label", f"Whisper {a}"),
            "engine": "faster-whisper",
            "installed": bool(entry.get("installed")),
            "available": True,
            "size_mb": entry.get("size_mb", 0),
            "install_state": entry.get("install_state", "idle"),
            "install_progress": entry.get("install_progress", 0),
            "downloadable": bool(entry),
            "has_source": bool(entry),
        })
    # Remaining catalog entries (upscale/VLM, plus any audio id not in
    # reg["audio_models"]) that are not already listed.
    known = {d["id"] for d in data}
    for m in cat.values():
        if m.get("id") in known:
            continue
        data.append({
            "id": m["id"], "object": "model", "owned_by": "unified",
            "kind": m.get("kind", "image"),
            "label": m.get("label", m["id"]),
            "engine": m.get("engine", ""),
            "installed": bool(m.get("installed")),
            "available": bool(m.get("installed")),
            "size_mb": m.get("size_mb", 0),
            "install_state": m.get("install_state", "idle"),
            "install_progress": m.get("install_progress", 0),
        })
    return {
        "object": "list",
        "data": data,
        "currently_loaded_model": st["loaded_id"],
        "loading": st["loading"],
        "loading_model": st["loading_model"],
        "missing_models": [k for k, v in models.items() if v.get("missing")],
        "default_image_model": reg.get("default_image_model", ""),
        "default_audio_model": reg.get("default_audio_model", "small"),
        "audio_models": reg.get("audio_models", []),
    }


@app.get("/v1/images/progress/{request_id}")
def image_progress(request_id: str):
    try:
        import diffusion_server as ds
        prog = getattr(ds, "_PROGRESS", {}).get(request_id)
        if prog:
            return {"id": request_id, **prog}
    except Exception:
        pass
    return {"id": request_id, "status": "unknown", "step": 0, "total": 0, "percent": 0}


# ---------- images ----------

def _resolve_image_model(requested):
    reg = _image_registry()
    if requested and requested in reg.get("image_models", {}):
        return requested
    return reg.get("default_image_model", "") or (requested or "")


def _try_real_generation(model_id, prompt, size, quality, negative_prompt="", guidance=1.0, seed=1):
    """Try torch diffusers, then the ONNX LCM pipeline.

    Returns (b64, engine) so the response never hides which engine produced the
    pixels; (None, "") means the caller has to use a placeholder.
    """
    last_err = ""
    try:
        import diffusion_server as ds
        req_id = f"unified-{uuid.uuid4().hex[:8]}"
        req = ds.ImageRequest(model=model_id, prompt=prompt, size=size,
                              quality=quality, n=1, request_id=req_id)
        # ensure_model_loaded does its own lock + 503 on load; just delegate here
        pixels = 0
        try:
            w, h = size.lower().split("x")
            pixels = int(w) * int(h)
        except Exception:
            pass
        ds.ensure_model_loaded(model_id, pixels)
        out = ds._generate_locked(req)
        imgs = out.get("data", [])
        if imgs and imgs[0].get("b64_json"):
            return imgs[0]["b64_json"], "diffusers"
    except Exception as e:
        last_err = str(e)[:160]
        logger.warning(f"real generation fallback: {last_err}")

    # ONNX LCM fallback (no torch/diffusers required)
    try:
        from unified.lcm_onnx_numpy import find_lcm_dir, generate_b64
        d = find_lcm_dir()
        if d is not None:
            try:
                w, h = size.lower().split("x")
                side = max(256, min(512, max(int(w), int(h))))
            except Exception:
                side = 512
            q = (quality or "medium").lower()
            steps = 8 if q == "high" else (6 if q == "medium" else 4)
            return generate_b64(d, prompt, negative_prompt=negative_prompt,
                                guidance=float(guidance or 1.0), size=side,
                                steps=steps, seed=seed), "onnx-lcm"
    except Exception as e:
        last_err = f"{last_err} | onnx: {str(e)[:120]}"
        logger.warning(f"onnx generation fallback failed: {str(e)[:200]}")
    return None, (f"failed: {last_err}" if last_err else "")


@app.post("/v1/images/generations")
def generations(req: GenRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    model_id = _resolve_image_model(req.model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    _release_vlm_best_effort()
    n = max(1, min(4, req.n or 1))
    data = []
    engines = []
    for i in range(n):
        b64, engine = _try_real_generation(model_id, req.prompt, req.size or "768x768",
                                           req.quality or "medium",
                                           req.negative_prompt, req.guidance,
                                           seed=int(time.time()) + i * 7919)
        if not b64:
            b64 = _placeholder_png_b64()
            engine = engine or "placeholder"
        data.append({"b64_json": b64})
        engines.append(engine)
    try:
        w, h = (req.size or "768x768").lower().split("x")
        side = max(256, min(512, max(int(w), int(h))))
    except Exception:
        side = 512
    q = (req.quality or "medium").lower()
    steps = 8 if q == "high" else (6 if q == "medium" else 4)
    # Report the size of the image that was actually produced (the torch path
    # honours the request, the ONNX fallback generates a square at `side`).
    real_size = f"{side}x{side}"
    try:
        from PIL import Image as _Image
        _im = _Image.open(io.BytesIO(base64.b64decode(data[0]["b64_json"])))
        real_size = f"{_im.width}x{_im.height}"
    except Exception:
        pass
    stored = []
    for item in data:
        meta = _store_b64("image", ".png", item.get("b64_json", ""), rid)
        if meta:
            stored.append(meta)
            item["output"] = meta
    return {
        "created": int(time.time()),
        "data": data,
        "outputs": stored,
        "request_id": rid,
        "size": real_size,
        "requested_size": req.size or "768x768",
        "quality": q,
        "steps": steps,
        "engine": engines[0] if len(set(engines)) == 1 else "mixed",
        "engines": engines,
    }


@app.post("/v1/images/edits")
async def edits(
    image: UploadFile = File(...),
    prompt: str = Form(default=""),
    model: str = Form(default=""),
    n: int = Form(default=1),
    size: str = Form(default="768x768"),
    quality: str = Form(default="medium"),
    strength: float = Form(default=0.75),
    request_id: str = Form(default=""),
):
    rid = request_id or f"req-{uuid.uuid4().hex[:8]}"
    model_id = _resolve_image_model(model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    _release_vlm_best_effort()
    # img2img: prefer the selected diffusion model (honours the requested size
    # and actually uses the model the user picked), then the ONNX LCM engine.
    raw = await image.read()
    b64 = None
    engine = None
    if raw:
        try:
            import diffusion_server as ds
            try:
                _w, _h = size.lower().split("x")
                _pixels = int(_w) * int(_h)
            except Exception:
                _pixels = 0
            with ds._tracked_model_lock():
                ds.ensure_model_loaded(model_id, _pixels)
                res = await ds._run_edit(prompt or "edit", raw, 1, size, quality,
                                         strength, "b64_json", rid)
            payload = (res.get("data") or [{}])[0].get("b64_json")
            if payload:
                b64 = payload
                engine = "diffusers"
            elif res.get("error"):
                logger.warning(f"diffusers edit unavailable: {str(res.get('error'))[:120]}")
        except Exception as e:
            logger.warning(f"diffusers edit failed: {str(e)[:160]}")
    try:
        from unified.lcm_onnx_numpy import find_lcm_dir, img2img_b64
        d = find_lcm_dir()
        if b64 is None and d is not None and raw:
            import base64 as _b64
            try:
                w, h = size.lower().split("x")
                side = max(256, min(512, max(int(w), int(h))))
            except Exception:
                side = 512
            steps = 6 if (quality or "medium") == "medium" else 4
            b64 = img2img_b64(d, _b64.b64encode(raw).decode(), prompt or "edit",
                              strength=float(strength or 0.75), size=side, steps=steps)
            engine = "onnx-lcm"
    except Exception as e:
        logger.warning(f"onnx img2img failed: {str(e)[:200]}")
    if not b64:
        b64, engine = _try_real_generation(model_id, prompt or "edit", size, quality)
    if not b64:
        b64 = _placeholder_png_b64(color=(30, 120, 200))
        engine = "placeholder"
    real_size = ""
    try:
        from PIL import Image as _Image
        _im = _Image.open(io.BytesIO(base64.b64decode(b64)))
        real_size = f"{_im.width}x{_im.height}"
    except Exception:
        pass
    meta = _store_b64("image", ".png", b64, rid)
    return {"created": int(time.time()), "data": [{"b64_json": b64}],
            "request_id": rid, "engine": engine, "size": real_size,
            "outputs": [meta] if meta else []}


@app.post("/v1/images/inpaint")
def inpaint(req: InpaintRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    model_id = _resolve_image_model(req.model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    t0 = time.time()
    _release_vlm_best_effort()
    # One big engine at a time on a 6 GB card: a resident diffusers pipeline
    # starved the ONNX inpaint and the request hung for minutes.
    _release_sd_best_effort()
    # Real pipeline when the engine is loaded (masks = removing people in the background),
    # otherwise echo the input — the contract {image, elapsed, request_id} always holds.
    if req.mask:
        real = None
        try:
            from unified.lcm_onnx_numpy import find_lcm_dir, inpaint_b64
            d = find_lcm_dir()
            if d is not None:
                try:
                    w, h = (req.width, req.height) if (req.width and req.height) else (512, 512)
                    side = max(256, min(512, max(int(w), int(h))))
                except Exception:
                    side = 512
                steps = req.steps or 6
                real = inpaint_b64(d, req.image, req.mask, req.prompt,
                                   negative_prompt=req.negative_prompt,
                                   guidance=float(req.guidance or 1.0),
                                   strength=float(req.strength or 0.8),
                                   size=side, steps=steps,
                                   mode=req.mode or "add", color=req.color or "#0066ff")
                meta = _store_b64("image", ".png", real, rid)
                return {"image": real, "elapsed": round(time.time() - t0, 2),
                        "request_id": rid, "engine": "onnx-inpaint",
                        "outputs": [meta] if meta else []}
        except Exception as e:
            logger.warning(f"onnx inpaint failed: {str(e)[:200]}")
        try:
            real = try_real_inpaint(req.image, req.mask, req.prompt, model_id,
                                    req.strength, req.steps)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)[:200]})
        except Exception as e:
            logger.warning(f"real inpaint failed, echo fallback: {str(e)[:160]}")
            real = None
        if real:
            meta = _store_b64("image", ".png", real, rid)
            return {"image": real, "elapsed": round(time.time() - t0, 2),
                    "request_id": rid, "engine": "diffusion",
                    "outputs": [meta] if meta else []}
    return {"image": req.image, "elapsed": round(time.time() - t0, 2),
            "request_id": rid, "engine": "echo"}


MAX_OUTPAINT_SIDE = 2048
MAX_OUTPAINT_PIXELS = 4_000_000


def _edge_pad_image(img, left: int, right: int, top: int, bottom: int):
    """Pad with replicated edge pixels so the model has real context to extend."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(img.convert("RGB"))
    padded = np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="edge")
    return Image.fromarray(padded, "RGB")


def _border_mask(width: int, height: int, left: int, right: int,
                 top: int, bottom: int):
    import numpy as np
    from PIL import Image

    m = np.zeros((height, width), dtype=np.uint8)
    if left:
        m[:, :left] = 255
    if right:
        m[:, width - right:] = 255
    if top:
        m[:top, :] = 255
    if bottom:
        m[height - bottom:, :] = 255
    return Image.fromarray(m, "L")


@app.post("/v1/images/outpaint")
def outpaint(req: OutpaintRequest):
    """Expand the canvas and let the inpaint model paint the new border."""
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    model_id = _resolve_image_model(req.model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    try:
        import numpy as np  # noqa: F401
        from unified.lcm_onnx_numpy import (_b64_image_to_pil, _pil_to_b64,
                                            find_lcm_dir, inpaint_b64)
    except Exception as e:
        return JSONResponse(status_code=503, content={
            "error": "will not load", "detail": f"outpaint engine unavailable: {str(e)[:160]}"})
    d = find_lcm_dir()
    if d is None:
        return JSONResponse(status_code=503, content={
            "error": "will not load",
            "detail": "LCM ONNX pipeline not found; install LCM_Dreamshaper_ONNX"})

    try:
        src = _b64_image_to_pil(req.image).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad image b64: {str(e)[:120]}"})

    w, h = src.size
    short = min(w, h)
    left = int(req.left or 0)
    right = int(req.right or 0)
    top = int(req.top or 0)
    bottom = int(req.bottom or 0)
    if req.margin and req.margin > 0:
        m = int(round(short * float(req.margin)))
        left = left or m
        right = right or m
        top = top or m
        bottom = bottom or m
    if min(left, right, top, bottom) < 0:
        return JSONResponse(status_code=400, content={"error": "negative margin"})
    if not any((left, right, top, bottom)):
        return JSONResponse(status_code=400, content={
            "error": "nothing to expand: pass left/right/top/bottom pixels or margin"})
    new_w, new_h = w + left + right, h + top + bottom
    if new_w > MAX_OUTPAINT_SIDE or new_h > MAX_OUTPAINT_SIDE:
        return JSONResponse(status_code=400, content={
            "error": f"result too large: {new_w}x{new_h} (max side {MAX_OUTPAINT_SIDE})"})
    if new_w * new_h > MAX_OUTPAINT_PIXELS:
        return JSONResponse(status_code=400, content={
            "error": f"result too large: {new_w}x{new_h} (max {MAX_OUTPAINT_PIXELS} pixels)"})

    t0 = time.time()
    _release_sd_best_effort()
    padded = _edge_pad_image(src, left, right, top, bottom)
    mask = _border_mask(new_w, new_h, left, right, top, bottom)
    size = max(256, min(512, max(new_w, new_h)))
    try:
        out_b64 = inpaint_b64(d, _pil_to_b64(padded), _pil_to_b64(mask),
                              req.prompt, negative_prompt=req.negative_prompt,
                              guidance=float(req.guidance or 1.0),
                              strength=float(req.strength or 0.99),
                              size=int(size), steps=int(req.steps or 6),
                              mode="add")
    except Exception as e:
        logger.warning(f"outpaint failed: {str(e)[:200]}")
        return JSONResponse(status_code=500, content={"error": f"outpaint failed: {str(e)[:200]}"})
    meta = _store_b64("image", ".png", out_b64, rid)
    return {"image": out_b64, "elapsed": round(time.time() - t0, 2),
            "outputs": [meta] if meta else [],
            "request_id": rid, "engine": "onnx-outpaint",
            "input_size": [w, h], "output_size": [new_w, new_h],
            "margins": {"left": left, "right": right, "top": top, "bottom": bottom}}


@app.post("/v1/images/harmonize")
def harmonize(req: HarmonizeRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    model_id = _resolve_image_model(req.model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    t0 = time.time()
    mask = req.mask or req.body_mask or req.seam_mask or ""
    try:
        from unified.lcm_onnx_numpy import harmonize_b64
        out = harmonize_b64(req.image, mask, color_match=req.color_match,
                            seam_fix=req.seam_fix, strength=req.strength)
        return {"image": out, "elapsed": round(time.time() - t0, 2),
                "request_id": rid, "engine": "cpu-harmonize"}
    except Exception as e:
        logger.warning(f"harmonize failed: {str(e)[:200]}")
        return {"image": req.image, "elapsed": round(time.time() - t0, 2),
                "request_id": rid, "engine": "echo"}


@app.post("/v1/images/upscale")
def upscale(req: UpscaleRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    try:
        _manager.ensure("upscale", "RealESRGAN_x4v3")
    except ModelLoadingError as e:
        return _raise_loading(e)
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    _release_sd_best_effort()
    try:
        out = upscale_image_b64(req.image, req.scale, req.model or "RealESRGAN_x4v3")
    except ValueError as e:
        msg = str(e)
        if "exceeds cap" in msg:
            return JSONResponse(status_code=503, content={"error": "will not load", "detail": msg})
        return JSONResponse(status_code=400, content={"error": msg})
    out["request_id"] = rid
    meta = _store_b64("image", ".png", out.get("image", ""), rid)
    out["outputs"] = [meta] if meta else []
    return out


# ---------- vision (V2) ----------

@app.post("/v1/images/caption")
def caption(req: CaptionRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    if _engines_disabled():
        return JSONResponse(status_code=503, content={
            "error": "will not load", "detail": "engines disabled by UNIFIED_NO_ENGINE"})
    try:
        _manager.ensure("vlm", VLM_ID)
    except ModelLoadingError as e:
        return _raise_loading(e)
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    _release_sd_best_effort()
    try:
        out = caption_image_b64(req.image, req.prompt or "Describe the image in detail.",
                                req.max_tokens or 256)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except EngineMissing as e:
        return JSONResponse(status_code=503,
                            content={"error": "will not load", "detail": str(e)[:200]})
    out["request_id"] = rid
    try:
        meta = _store_output("text", ".txt", str(out.get("caption", "")).encode("utf-8"), rid)
        out["outputs"] = [meta]
    except Exception:
        out["outputs"] = []
    return out


def _chat_vision_parts(messages):
    """Extract (text, image_b64) from the last user message in OpenAI messages."""
    text, image = "", ""
    items = messages if isinstance(messages, list) else []
    for m in reversed(items):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            if c.strip():
                text = c
                break
        elif isinstance(c, list):
            for part in c:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and not text:
                    text = part.get("text", "")
                elif part.get("type") == "image_url" and not image:
                    u = (part.get("image_url") or {}).get("url", "")
                    if u.startswith("data:image/") and ";base64," in u:
                        image = u.split(";base64,", 1)[1]
            if text or image:
                break
    return text, image


@app.post("/v1/chat/completions")
def chat_completions(body: dict):
    model = (body.get("model") or VLM_ID) if isinstance(body, dict) else VLM_ID
    messages = body.get("messages", []) if isinstance(body, dict) else []
    max_tokens = body.get("max_tokens", 256) if isinstance(body, dict) else 256
    if _engines_disabled():
        return JSONResponse(status_code=503, content={
            "error": "will not load", "detail": "engines disabled by UNIFIED_NO_ENGINE"})
    try:
        _manager.ensure("vlm", VLM_ID)
    except ModelLoadingError as e:
        return _raise_loading(e)
    text, image = _chat_vision_parts(messages)
    if not image:
        return JSONResponse(status_code=400, content={
            "error": "missing image: V1 supports only vision messages (content with image_url data:base64)"})
    _release_sd_best_effort()
    try:
        out = caption_image_b64(image, text or "Describe the image in detail.",
                                max_tokens or 256)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except EngineMissing as e:
        return JSONResponse(status_code=503,
                            content={"error": "will not load", "detail": str(e)[:200]})
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": out["text"]},
            "finish_reason": "stop",
        }],
    }


# ---------- audio ----------

# ---------- job progress / cancellation ----------
# Long-running work (transcription today, more kinds later) registers here so a
# client can poll /v1/jobs/{id} and stop it with /v1/jobs/{id}/cancel.
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_JOB_HISTORY = 50


class JobCancelled(Exception):
    pass


def _job_start(rid: str, kind: str, total: float = 0.0, message: str = "") -> None:
    with _JOBS_LOCK:
        _JOBS[rid] = {
            "id": rid, "kind": kind, "state": "running", "percent": 0,
            "done": 0.0, "total": float(total or 0), "message": message,
            "started": time.time(), "cancel": False, "error": "",
        }
        if len(_JOBS) > _JOB_HISTORY:
            for old in sorted(_JOBS, key=lambda k: _JOBS[k]["started"])[:10]:
                if old != rid and _JOBS[old]["state"] != "running":
                    _JOBS.pop(old, None)


def _job_update(rid: str, done: float = None, total: float = None,
                message: str = None) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(rid)
        if not job:
            return
        if done is not None:
            job["done"] = float(done)
        if total is not None:
            job["total"] = float(total)
        if message is not None:
            job["message"] = message
        total = job["total"]
        if total > 0:
            job["percent"] = int(max(0, min(100, 100.0 * job["done"] / total)))


def _job_finish(rid: str, state: str = "done", error: str = "") -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(rid)
        if not job:
            return
        job["state"] = state
        job["error"] = error[:300]
        job["percent"] = 100 if state == "done" else job.get("percent", 0)
        job["finished"] = time.time()


def _job_cancelled(rid: str) -> bool:
    with _JOBS_LOCK:
        job = _JOBS.get(rid)
        return bool(job and job.get("cancel"))


def _job_snapshot(rid: str) -> dict:
    with _JOBS_LOCK:
        job = _JOBS.get(rid)
        if not job:
            return {"id": rid, "state": "unknown", "percent": 0}
        snap = dict(job)
    if snap.get("started"):
        end = snap.get("finished") or time.time()
        snap["elapsed"] = round(end - snap["started"], 1)
    return snap


# ---------- job output store ----------
# Every produced artifact is written to pc/output/<kind>/ so a job leaves a file
# behind instead of only a base64 blob in the response.
OUTPUT_DIR = HERE / "output"
OUTPUT_KINDS = ("image", "video", "audio", "text")


def _store_output(kind: str, ext: str, data: bytes, rid: str = "", stem: str = "") -> dict:
    """Write one artifact under output/<kind>/ and return its descriptor."""
    kind = kind if kind in OUTPUT_KINDS else "text"
    ext = ext if ext.startswith(".") else f".{ext}"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_rid = "".join(c for c in (rid or "") if c.isalnum() or c in "-_")[:32].strip("-_")
    base = stem or f"{stamp}-{safe_rid or uuid.uuid4().hex[:8]}"
    target = OUTPUT_DIR / kind
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{base}{ext}"
    n = 1
    while path.exists():
        path = target / f"{base}-{n}{ext}"
        n += 1
    path.write_bytes(data)
    return {
        "name": path.name,
        "kind": kind,
        "size": len(data),
        "path": f"{kind}/{path.name}",
        "url": f"/v1/outputs/{kind}/{path.name}",
    }


def _store_b64(kind: str, ext: str, b64: str, rid: str = "") -> dict:
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return {}
    return _store_output(kind, ext, raw, rid)


ALLOWED_AUDIO_EXT = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav",
                     ".webm", ".ogg", ".flac", ".opus"}
MAX_FILE_SIZE = 500 * 1024 * 1024
ALLOWED_WHISPER = ("tiny", "base", "small", "medium", "turbo", "large")

_WHISPER_CACHE = {}


_AV_PATCHED = False


def _patch_av_compat():
    """PyAV 19 removed av.open(metadata_errors=...), which faster-whisper
    still passes. Drop the keyword instead of failing the request."""
    global _AV_PATCHED
    if _AV_PATCHED:
        return
    _AV_PATCHED = True
    try:
        import av
        orig = av.open
        def _open(*args, **kwargs):
            kwargs.pop("metadata_errors", None)
            return orig(*args, **kwargs)
        av.open = _open
    except Exception:
        pass


def _get_whisper_model(source, device: str, compute: str):
    """One loaded faster-whisper model per (source, device, compute)."""
    key = (str(source), device, compute)
    m = _WHISPER_CACHE.get(key)
    if m is None:
        _patch_av_compat()
        from faster_whisper import WhisperModel
        m = WhisperModel(str(source), device=device, compute_type=compute)
        _WHISPER_CACHE[key] = m
    return m


def release_whisper_cache() -> bool:
    had = bool(_WHISPER_CACHE)
    _WHISPER_CACHE.clear()
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    return had


def _transcribe_with_whisper(tmp_path: str, wanted: str, language, rid: str = "",
                             initial_prompt: str = ""):
    # CTranslate2 4.x needs the CUDA 12 runtime (cublas64_12.dll/cudnn64_9.dll).
    # Default to CPU int8 so transcription works without those DLLs; set
    # WHISPER_DEVICE=cuda when the CUDA 12 libraries are installed.
    device = (os.environ.get("WHISPER_DEVICE") or "cpu").strip().lower()
    if device not in ("cpu", "cuda"):
        device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    # Prefer a locally downloaded CTranslate2 model (offline, no HF CDN):
    #   pc/models/faster-whisper-<name>/model.bin
    # whisper_installed() checks the exact expected size + a valid config.json,
    # so a half-finished download is never loaded as if it were complete.
    local = Path(__file__).resolve().parent / "models" / f"faster-whisper-{wanted}"
    try:
        from unified.models_manager import whisper_installed
        use_local = whisper_installed(wanted)
    except Exception:
        local_bin = local / "model.bin"
        use_local = local_bin.is_file() and local_bin.stat().st_size > 50 * 1024 * 1024
    if not use_local:
        local_bin = local / "model.bin"
        if local_bin.is_file():
            logger.warning(
                f"whisper '{wanted}' is not fully downloaded "
                f"({local_bin.stat().st_size} B); will use the HF download")
    source = str(local) if use_local else wanted

    def _run(model, rid: str = "", duration: float = 0.0) -> tuple[str, str, list]:
        kwargs = {"language": language, "beam_size": 1, "vad_filter": True}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        segs, info = model.transcribe(tmp_path, **kwargs)
        items = []
        for s in segs:
            if rid and _job_cancelled(rid):
                raise JobCancelled()
            txt = (s.text or "").strip()
            if not txt:
                continue
            if rid:
                _job_update(rid, done=float(s.end),
                            message=f"segment {len(items) + 1}")
            items.append({"start": round(float(s.start), 2),
                          "end": round(float(s.end), 2),
                          "text": txt})
        text = " ".join(i["text"] for i in items).strip()
        return text, getattr(info, "language", language), items

    try:
        return _run(_get_whisper_model(source, device, compute), rid)
    except JobCancelled:
        raise
    except Exception as e:
        if device == "cuda":
            logger.warning(f"whisper CUDA failed ({str(e)[:140]}); retrying on CPU int8")
            return _run(_get_whisper_model(source, "cpu", "int8"), rid)
        raise


def _segments_to_srt(segments: list) -> str:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        sec, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{ts(seg['start'])} --> {ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _segments_to_vtt(segments: list) -> str:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        sec, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{sec:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{ts(seg['start'])} --> {ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _transcribe_chunked(audio_path: Path, wanted: str, language, rid: str = ""):
    """Long file -> 5-minute chunks: gives real progress and a working cancel.

    faster-whisper decodes the whole file before yielding the first segment, so
    a single pass cannot report progress or be interrupted. Chunking with the
    tail of the previous text as `initial_prompt` keeps the context.
    """
    try:
        dur = float(get_audio_duration(audio_path) or 0.0)
    except Exception:
        dur = 0.0
    if dur <= 600 or _job_cancelled(rid):
        if rid and _job_cancelled(rid):
            raise JobCancelled()
        return _transcribe_with_whisper(str(audio_path), wanted, language, rid)

    from unified.audio_splitter import split_audio_with_ffmpeg, cleanup_chunks
    chunks = split_audio_with_ffmpeg(audio_path, chunk_seconds=300, overlap_seconds=0,
                                     output_dir=Path(tempfile.gettempdir()))
    _job_update(rid, total=dur, message=f"{len(chunks)} chunks")
    texts, segments, prev = [], [], ""
    try:
        for idx, (chunk_path, start, end) in enumerate(chunks):
            if _job_cancelled(rid):
                raise JobCancelled()
            text, _lang, segs = _transcribe_with_whisper(
                str(chunk_path), wanted, language, "", initial_prompt=prev[-300:])
            for seg in segs:
                segments.append({"start": round(start + seg["start"], 2),
                                 "end": round(min(start + seg["end"], end), 2),
                                 "text": seg["text"]})
            texts.append(text)
            prev = (prev + " " + text).strip()[-400:]
            _job_update(rid, done=end, message=f"chunk {idx + 1}/{len(chunks)}")
    finally:
        cleanup_chunks(chunks, keep_original=True)
    return " ".join(texts).strip(), language, segments


async def _transcribe_with_splitting(audio_path: Path, wanted: str, language,
                                    rid: str = ""):
    """Transcribe long audio.

    faster-whisper handles long files with its own VAD, so splitting is only
    used for the remote WhisperApi fallback (and for legacy engines).
    """
    def _transcribe_single(path: Path, lang: Optional[str]) -> Tuple[str, str, list]:
        return _transcribe_with_whisper(str(path), wanted, lang, rid)

    try:
        import faster_whisper  # noqa: F401
        return _transcribe_single(audio_path, language)
    except ImportError:
        pass

    try:
        # Use the audio splitter utility (remote service path): text only.
        from unified.audio_splitter import transcribe_with_splitting
        text, lang = await transcribe_with_splitting(audio_path, _transcribe_single, language)
        return text, lang, []
    except Exception as e:
        logger.warning(f"Audio splitting failed, falling back to single transcribe: {e}")
        return _transcribe_single(audio_path, language)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(default=None),
    audio: UploadFile = File(default=None),
    url: str = Form(default=""),
    model: str = Form(default="whisper-1"),
    model_name: str = Form(default="whisper-1"),
    device: str = Form(default=""),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
    auto_split: str = Form(default="true"),
    request_id: str = Form(default=""),
):
    # Called directly (translations) FastAPI defaults are still Form objects.
    if not isinstance(request_id, str):
        request_id = ""
    rid = request_id or f"req-{uuid.uuid4().hex[:8]}"
    up = file or audio
    url = (url or "").strip()
    if url:
        if up is not None:
            return JSONResponse(status_code=400, content={"error": "send either file or url, not both"})
        try:
            from unified.ytdlp import download_audio, YtDlpMissing, DownloadFailed
            tmp_wav = download_audio(url)
        except YtDlpMissing as e:
            return JSONResponse(status_code=503, content={"error": "yt-dlp missing",
                                                         "detail": str(e)[:200]})
        except DownloadFailed as e:
            return JSONResponse(status_code=400, content={"error": "download failed",
                                                         "detail": str(e)[:200]})
        content = Path(tmp_wav).read_bytes()
        filename = Path(tmp_wav).name
        suffix = ".wav"
        up = None
    if up is None and not url:
        return JSONResponse(status_code=400, content={"error": "missing file or url"})
    raw_model = (model or "").strip() or (model_name or "").strip() or "whisper-1"
    wanted = "small" if raw_model == "whisper-1" else raw_model
    if wanted not in ALLOWED_WHISPER:
        return JSONResponse(status_code=400,
                            content={"error": f"Unknown model '{raw_model}'"})
    if device and device.strip().lower() not in ("cuda", "cpu", ""):
        return JSONResponse(status_code=400, content={"error": "Unknown device"})
    import os
    suffix = os.path.splitext(up.filename or "")[1].lower() or ".wav"
    if suffix not in ALLOWED_AUDIO_EXT:
        return JSONResponse(status_code=415, content={"error": f"Unsupported audio format: {suffix}"})
    content = await up.read() if up is not None else content
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file uploaded"})
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(status_code=413, content={"error": "File too large"})
    try:
        _manager.ensure("audio", wanted)
    except ModelLoadingError as e:
        return _raise_loading(e)
    _job_start(rid, "transcribe", 0.0, f"model={wanted}")
    logger.info(f"transcribe: {len(content)} bytes, model={wanted}, lang={language}")
    import tempfile
    tmp = None
    segments = []
    duration_hint = 0.0
    try:
        try:
            duration_hint = float(get_audio_duration(Path(tmp)) or 0.0) if tmp else 0.0
        except Exception:
            duration_hint = 0.0
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)
        try:
            _job_update(rid, total=float(get_audio_duration(Path(tmp)) or 0.0))
        except Exception:
            pass
        try:
            split_enabled = auto_split.lower() in ("true", "1", "yes", "on")
            logger.info(f"transcribe: starting engine (split={split_enabled})")
            if split_enabled:
                text, lang, segments = await _transcribe_with_splitting(
                    Path(tmp), wanted, language, rid)
            else:
                text, lang, segments = await asyncio.to_thread(
                    _transcribe_chunked, Path(tmp), wanted, language, rid)
            engine = "local"
        except JobCancelled:
            # Must not fall through to the proxy: the user asked to stop.
            raise
        except Exception as e:
            logger.warning("local whisper missing (%s), proxy %s",
                             str(e)[:120], WHISPER_SERVICE_URL)
            proxied = transcribe_via_service(
                WHISPER_SERVICE_URL, content, up.filename or "audio.wav",
                wanted, language)
            if proxied is not None:
                text, lang = proxied
                segments = []
                engine = "whisper-service"
                # nothing is kept in RAM locally → release the gate immediately for images
                _manager.unload("audio-proxied")
            else:
                text, lang = "[unified stub — start the WhisperApi service or install faster-whisper]", language
                segments = []
                engine = "stub"
        _job_finish(rid, "done")
        fmt = (response_format or "json").strip().lower()
        if fmt in ("srt", "vtt"):
            body = _segments_to_srt(segments) if fmt == "srt" else _segments_to_vtt(segments)
            if not segments:
                # no timings (stub/proxy) -> one cue covering the whole text
                dur = get_audio_duration(Path(tmp)) or 0.0
                cue = [{"start": 0.0, "end": max(dur, 1.0), "text": text}]
                body = _segments_to_srt(cue) if fmt == "srt" else _segments_to_vtt(cue)
            meta = _store_output("text", f".{fmt}", body.encode("utf-8"), rid)
            from fastapi.responses import PlainTextResponse
            resp = PlainTextResponse(body)
            resp.headers["X-Output-Path"] = meta["path"]
            return resp
        if fmt == "verbose_json":
            meta = _store_output("text", ".txt", text.encode("utf-8"), rid)
            return {"text": text, "language": lang, "engine": engine,
                    "duration": get_audio_duration(Path(tmp)),
                    "segments": segments, "outputs": [meta]}
        if fmt == "text":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(text)
        meta = _store_output("text", ".txt", text.encode("utf-8"), rid)
        return {"text": text, "language": lang, "engine": engine,
                "segments": segments, "outputs": [meta]}
    except JobCancelled:
        _job_finish(rid, "cancelled")
        return JSONResponse(status_code=409, content={
            "error": "cancelled", "request_id": rid})
    except Exception as e:
        _job_finish(rid, "error", str(e))
        raise
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/v1/audio/translations")
async def translations(
    file: UploadFile = File(default=None),
    audio: UploadFile = File(default=None),
    model: str = Form(default="whisper-1"),
):
    r = await transcriptions(file=file, audio=audio, url="", model=model,
                             model_name=model, device="", language=None,
                             response_format="json", auto_split="true",
                             request_id="")
    if isinstance(r, dict) and "text" in r:
        return {"text": r["text"]}
    return r


@app.post("/transcribe")
async def transcribe_alias(file: UploadFile = File(...), language: str = Form(default=None)):
    return await transcriptions(file=file, language=language)


# ---------- webui ----------

@app.get("/webui", response_class=HTMLResponse)
def webui():
    for name in ("webui_unified.html", "webui.html"):
        p = HERE / name
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    return ('<h3>Unified Standalone</h3><ul>'
            '<li>POST /v1/images/generations|edits|inpaint|harmonize|upscale</li>'
            '<li>POST /v1/audio/transcriptions|translations</li></ul>')




# ---------- ffmpeg tools ----------

class ConvertRequest(BaseModel):
    preset: str = "mp4-h264"
    resize: Optional[str] = None
    trim_start: Optional[float] = None
    trim_end: Optional[float] = None
    request_id: Optional[str] = ""


class AudioConvertRequest(BaseModel):
    preset: str = "mp3-128"
    trim_start: Optional[float] = None
    trim_end: Optional[float] = None
    request_id: Optional[str] = ""


class TrimRequest(BaseModel):
    start: float = 0.0
    end: float = 10.0
    request_id: Optional[str] = ""


@app.get("/v1/ffmpeg/presets")
def ffmpeg_presets():
    return get_presets()


@app.post("/v1/ffmpeg/convert-video")
async def convert_video_endpoint(
    file: UploadFile = File(...),
    preset: str = Form(default="mp4-h264"),
    resize: str = Form(default=""),
    trim_start: str = Form(default=""),
    trim_end: str = Form(default=""),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(status_code=413, content={"error": "File too large"})

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        ts = float(trim_start) if trim_start.strip() else None
        te = float(trim_end) if trim_end.strip() else None

        out_path, info = convert_video(
            Path(tmp), preset=preset,
            resize=resize.strip() or None,
            trim_start=ts, trim_end=te
        )

        out_data = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(out_data).decode()
        _ext = Path(out_path).suffix.lower() or ".bin"
        _kind = ("image" if _ext in (".jpg", ".jpeg", ".png", ".webp", ".avif") else
                 "audio" if _ext in (".mp3", ".m4a", ".aac", ".wav", ".opus", ".flac") else
                 "video")
        out_meta = _store_output(_kind, _ext, out_data, rid)

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": Path(out_path).name,
            "elapsed": 0,
            "request_id": rid,
            "info": {
                "duration": info.get("format", {}).get("duration"),
                "size": info.get("format", {}).get("size"),
                "format": info.get("format", {}).get("format_name"),
            }
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/v1/ffmpeg/convert-audio")
async def convert_audio_endpoint(
    file: UploadFile = File(...),
    preset: str = Form(default="mp3-128"),
    trim_start: str = Form(default=""),
    trim_end: str = Form(default=""),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        ts = float(trim_start) if trim_start.strip() else None
        te = float(trim_end) if trim_end.strip() else None

        out_path, info = convert_audio(
            Path(tmp), preset=preset,
            trim_start=ts, trim_end=te
        )

        out_data = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(out_data).decode()
        _ext = Path(out_path).suffix.lower() or ".bin"
        _kind = ("image" if _ext in (".jpg", ".jpeg", ".png", ".webp", ".avif") else
                 "audio" if _ext in (".mp3", ".m4a", ".aac", ".wav", ".opus", ".flac") else
                 "video")
        out_meta = _store_output(_kind, _ext, out_data, rid)

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": Path(out_path).name,
            "request_id": rid,
            "info": {
                "duration": info.get("format", {}).get("duration"),
                "size": info.get("format", {}).get("size"),
                "format": info.get("format", {}).get("format_name"),
            }
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/v1/ffmpeg/convert-image")
async def convert_image_endpoint(
    file: UploadFile = File(...),
    preset: str = Form(default="jpg-95"),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})
    suffix = Path(file.filename or "image.png").suffix or ".png"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)
        out_path, info = convert_image(Path(tmp), preset=preset)
        _raw = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(_raw).decode()
        _ext = Path(out_path).suffix.lower() or ".jpg"
        out_meta = _store_output("image", _ext, _raw, rid)
        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass
        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": Path(out_path).name,
            "request_id": rid,
            "info": {
                "size": info.get("format", {}).get("size"),
                "format": info.get("format", {}).get("format_name"),
            },
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/v1/ffmpeg/extract-audio")
async def extract_audio_endpoint(
    file: UploadFile = File(...),
    format: str = Form(default="wav"),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        out_path, info = extract_audio(Path(tmp), audio_format=format)

        out_data = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(out_data).decode()
        _ext = Path(out_path).suffix.lower() or ".bin"
        _kind = ("image" if _ext in (".jpg", ".jpeg", ".png", ".webp", ".avif") else
                 "audio" if _ext in (".mp3", ".m4a", ".aac", ".wav", ".opus", ".flac") else
                 "video")
        out_meta = _store_output(_kind, _ext, out_data, rid)

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": Path(out_path).name,
            "request_id": rid,
            "info": {
                "duration": info.get("format", {}).get("duration"),
                "size": info.get("format", {}).get("size"),
            }
        }
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/v1/ffmpeg/trim")
async def trim_endpoint(
    file: UploadFile = File(...),
    start: float = Form(default=0.0),
    end: float = Form(default=10.0),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    suffix = Path(file.filename or "media.mp4").suffix or ".mp4"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        out_path, info = trim_media(Path(tmp), start, end)

        out_data = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(out_data).decode()
        _ext = Path(out_path).suffix.lower() or ".bin"
        _kind = ("image" if _ext in (".jpg", ".jpeg", ".png", ".webp", ".avif") else
                 "audio" if _ext in (".mp3", ".m4a", ".aac", ".wav", ".opus", ".flac") else
                 "video")
        out_meta = _store_output(_kind, _ext, out_data, rid)

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": Path(out_path).name,
            "request_id": rid,
            "info": {
                "duration": info.get("format", {}).get("duration"),
                "size": info.get("format", {}).get("size"),
            }
        }
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/v1/ffmpeg/compress")
async def compress_endpoint(
    file: UploadFile = File(...),
    target_size_mb: float = Form(default=10.0),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        out_path, info = compress_video(Path(tmp), target_size_mb)

        out_data = Path(out_path).read_bytes()
        out_b64 = base64.b64encode(out_data).decode()
        _ext = Path(out_path).suffix.lower() or ".bin"
        _kind = ("image" if _ext in (".jpg", ".jpeg", ".png", ".webp", ".avif") else
                 "audio" if _ext in (".mp3", ".m4a", ".aac", ".wav", ".opus", ".flac") else
                 "video")
        out_meta = _store_output(_kind, _ext, out_data, rid)

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
            "outputs": [out_meta],
            "filename": Path(out_path).name,
            "request_id": rid,
            "info": {
                "duration": info.get("format", {}).get("duration"),
                "size": info.get("format", {}).get("size"),
            }
        }
    except (ValueError, RuntimeError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)[:200]})
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/v1/ffmpeg/info")
async def media_info_endpoint(
    file: UploadFile = File(...),
    request_id: str = Form(default=""),
):
    rid = request_id or "req-" + uuid.uuid4().hex[:8]
    content = await file.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file"})

    suffix = Path(file.filename or "media.mp4").suffix or ".mp4"
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)

        info = get_media_info(Path(tmp))
        return {"info": info, "request_id": rid}
    finally:
        try:
            if tmp:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass




# ---------- jobs (progress / cancel) ----------

@app.get("/v1/jobs")
def list_jobs():
    with _JOBS_LOCK:
        items = [dict(j) for j in _JOBS.values()]
    items.sort(key=lambda j: j.get("started", 0), reverse=True)
    return {"count": len(items), "jobs": items[:50]}


@app.get("/v1/jobs/{rid}")
def get_job(rid: str):
    snap = _job_snapshot(rid)
    if snap.get("state") == "unknown":
        return JSONResponse(status_code=404, content={"error": "unknown job"})
    return snap


@app.post("/v1/jobs/{rid}/cancel")
def cancel_job(rid: str):
    with _JOBS_LOCK:
        job = _JOBS.get(rid)
        if not job:
            return JSONResponse(status_code=404, content={"error": "unknown job"})
        if job["state"] != "running":
            return {"id": rid, "state": job["state"], "cancelled": False}
        job["cancel"] = True
    logger.info("job %s: cancel requested", rid)
    return {"id": rid, "state": "cancelling", "cancelled": True}


# ---------- outputs ----------

def _output_path(rel: str) -> Path:
    """Resolve a client-supplied output path, refusing anything outside output/."""
    root = OUTPUT_DIR.resolve()
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise ValueError("path outside output/")
    return target


@app.get("/v1/outputs")
def list_outputs(limit: int = 200):
    files = []
    if OUTPUT_DIR.is_dir():
        for p in sorted(OUTPUT_DIR.rglob("*"), key=lambda x: x.stat().st_mtime,
                        reverse=True):
            if not p.is_file():
                continue
            rel = p.relative_to(OUTPUT_DIR).as_posix()
            st = p.stat()
            files.append({
                "name": p.name,
                "kind": p.parent.name,
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "path": rel,
                "url": f"/v1/outputs/{rel}",
            })
            if len(files) >= max(1, min(limit, 1000)):
                break
    return {"dir": str(OUTPUT_DIR), "count": len(files), "files": files}


@app.get("/v1/outputs/{rel:path}")
def get_output(rel: str):
    try:
        target = _output_path(rel)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    if not target.is_file():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(target, filename=target.name)


@app.delete("/v1/outputs/{rel:path}")
def delete_output(rel: str):
    try:
        target = _output_path(rel)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    if not target.is_file():
        return JSONResponse(status_code=404, content={"error": "not found"})
    target.unlink()
    return {"removed": rel}


@app.post("/v1/outputs/open")
def open_outputs():
    """Reveal the folder in the OS file manager (local desktop use only)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(OUTPUT_DIR))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(OUTPUT_DIR)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(OUTPUT_DIR)])
        return {"opened": str(OUTPUT_DIR)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"cannot open: {e}"})


# ---------- model install/uninstall (PC) ----------

from unified import models_manager as _mm


class InstallRequest(BaseModel):
    id: str = ""


@app.get("/v1/models/catalog")
def models_catalog():
    """Catalog of installable models + state (installed/downloading/…)."""
    try:
        return {
            "models": _mm.catalog(),
            "kinds": [
                {"id": k, "label": v} for k, v in _mm.KIND_LABELS.items()
            ],
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})


@app.post("/v1/models/install")
def models_install(req: InstallRequest):
    try:
        return _mm.start_install(req.id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})


@app.post("/v1/models/uninstall")
def models_uninstall(req: InstallRequest):
    try:
        return _mm.uninstall(req.id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})


@app.get("/v1/models/progress/{model_id}")
def models_progress(model_id: str):
    """Same shape as Android /v1/models/progress/{id}."""
    try:
        return _mm.progress(model_id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})


@app.get("/v1/models/status/{model_id}")
def models_status(model_id: str):
    try:
        return _mm.status(model_id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})


@app.post("/v1/models/verify/{model_id}")
def models_verify(model_id: str):
    try:
        return _mm.verify(model_id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})


@app.post("/v1/models/cancel")
def models_cancel(req: InstallRequest):
    try:
        return _mm.cancel(req.id)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})




@app.get("/v1/settings")
def settings_info(request: Request):
    """Runtime info for the WebUI Settings tab (read-only, no secrets)."""
    import shutil as _sh
    torch_ok = False
    cuda = False
    try:
        import torch
        torch_ok = True
        cuda = torch.cuda.is_available()
    except Exception:
        pass
    onnx_ok = False
    try:
        import onnxruntime  # noqa: F401
        onnx_ok = True
    except Exception:
        pass
    whisper_ok = False
    try:
        import faster_whisper  # noqa: F401
        whisper_ok = True
    except Exception:
        pass
    return {
        "version": "unified-0.1.0",
        # The port the request actually arrived on, not a hardcoded default.
        "port": request.url.port or 8101,
        "whisper_url": WHISPER_SERVICE_URL,
        "whisper_mode": "local (faster-whisper)" if whisper_ok else "remote",
        "models_dir": str(HERE / "models"),
        "output_dir": str(HERE / "output"),
        "default_image_model": _image_registry().get("default_image_model", ""),
        "audio_models": list(ALLOWED_WHISPER),
        "engines": {
            "torch": torch_ok,
            "cuda": cuda,
            "onnxruntime": onnx_ok,
            "whisper_native": whisper_ok,
            "ffmpeg": _sh.which("ffmpeg") or "",
            "ffprobe": _sh.which("ffprobe") or "",
            "ytdlp": _sh.which("yt-dlp") or "",
        },
    }




# ---------- model finder / custom models ----------

from unified import model_finder as _mf


class AddModelRequest(BaseModel):
    repo_id: str = ""
    name: str = ""
    url: str = ""
    kind: str = "custom"


class DeleteModelRequest(BaseModel):
    name: str = ""


@app.get("/v1/models/find")
def models_find(q: str = "", limit: int = 20, task: str = ""):
    try:
        return _mf.search_hf(q, limit=limit, filter_task=task)
    except RuntimeError as e:
        return JSONResponse(status_code=502, content={"error": str(e)[:300]})


@app.get("/v1/models/custom")
def models_custom():
    res = _mf.custom_models()
    for m in res["models"]:
        m.update(_mf.add_state(m["name"]))
    return res


@app.post("/v1/models/add")
def models_add(req: AddModelRequest):
    try:
        if req.url.strip():
            return _mf.start_add_url(req.url, req.name, req.kind)
        return _mf.start_add(req.repo_id, req.name)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/v1/models/add/status/{name}")
def models_add_status(name: str):
    return {"name": name, **_mf.add_state(name)}


@app.post("/v1/models/delete")
def models_delete(req: DeleteModelRequest):
    try:
        return _mf.delete_custom(req.name)
    except KeyError as e:
        return JSONResponse(status_code=404, content={"error": str(e).strip("'")})
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)[:200]})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unified_server:app", host="127.0.0.1", port=8101, workers=1)
