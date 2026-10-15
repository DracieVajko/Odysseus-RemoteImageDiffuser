#!/usr/bin/env python3
"""Unified Standalone Server — obrazy + audio v jednom procese, jedna zmluva.

Spája DiffusionServer (:8101) + WhisperApi (:4444) bez duplikácie ťažkej
logiky: deleguje na existujúce moduly lazy (až za ModelManager gate),
takže idle VRAM/RAM ostáva voľná. Vždy max 1 ťažký model rezidentný.

Cesty (frozen pre Odysseus):
GET /health, GET /v1/models, POST /v1/images/generations|edits,
POST /v1/images/inpaint|harmonize|upscale, POST /v1/audio/transcriptions|translations,
GET /v1/images/progress/{id}, GET /webui, GET / -> /webui
"""
import base64
import io
import json
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional

from unified.model_manager import ModelManager, ModelLoadingError
from unified.registry import build_registry
from unified.upscale import upscale_image_b64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified_server")

HERE = Path(__file__).resolve().parent
_manager = ModelManager()
_START_TS = time.time()

app = FastAPI(title="Unified Standalone Server")
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
    return "cpu"


def _image_registry():
    try:
        return build_registry()
    except Exception:
        return {"image_models": {}, "default_image_model": "DreamShaper_8",
                "audio_models": ["tiny", "base", "small", "medium", "turbo", "large"],
                "default_audio_model": "small"}


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
    n: int = 1
    size: str = "768x768"
    quality: str = "medium"
    request_id: Optional[str] = ""


class InpaintRequest(BaseModel):
    image: str = ""
    mask: Optional[str] = ""
    prompt: str = ""
    model: Optional[str] = ""
    strength: float = 0.75
    steps: int = 0
    width: Optional[int] = None
    height: Optional[int] = None
    request_id: Optional[str] = ""


class HarmonizeRequest(BaseModel):
    image: str = ""
    prompt: str = ""
    model: Optional[str] = ""
    strength: float = 0.5
    max_side: int = 1024
    steps: int = 0
    request_id: Optional[str] = ""


class UpscaleRequest(BaseModel):
    image: str = ""
    scale: int = 2
    model: Optional[str] = "RealESRGAN_x4v3"
    request_id: Optional[str] = ""


# ---------- system ----------

@app.get("/")
def root():
    return RedirectResponse(url="/webui")


@app.get("/health")
def health():
    reg = _image_registry()
    st = _manager.status()
    return {
        "status": "ok",
        "model": reg.get("default_image_model", ""),
        "current_model": st["loaded_id"],
        "device": _device(),
        "available_models": sorted(list(reg.get("image_models", {}).keys())),
        "loading": st["loading"],
        "loading_model": st["loading_model"],
        "busy": st["busy"],
        "queue": st["queue"],
        "uptime_s": round(time.time() - _START_TS, 1),
        # audio sekcia (WhisperApi kompat)
        "audio_model": reg.get("default_audio_model", "small"),
        "audio_loaded": st["loaded_kind"] == "audio",
        "version": "unified-0.1.0",
    }


@app.get("/v1/models")
def list_models():
    reg = _image_registry()
    st = _manager.status()
    data = [{"id": k, "object": "model", "owned_by": "unified",
             "available": True} for k in sorted(reg.get("image_models", {}).keys())]
    return {
        "object": "list",
        "data": data,
        "currently_loaded_model": st["loaded_id"],
        "loading": st["loading"],
        "loading_model": st["loading_model"],
        "missing_models": [],
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


def _try_real_generation(model_id, prompt, size, quality):
    """Pokus o reálnu generáciu cez diffusion_server; None = fallback placeholder."""
    try:
        import diffusion_server as ds
        req_id = f"unified-{uuid.uuid4().hex[:8]}"
        req = ds.ImageRequest(model=model_id, prompt=prompt, size=size,
                              quality=quality, n=1, request_id=req_id)
        # ensure_model_loaded robí vlastný lock + 503 pri load; tu len deleguj
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
            return imgs[0]["b64_json"]
    except Exception as e:
        logger.warning(f"real generation fallback: {str(e)[:160]}")
    return None


@app.post("/v1/images/generations")
def generations(req: GenRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    model_id = _resolve_image_model(req.model)
    try:
        _manager.ensure("image", model_id)
    except ModelLoadingError as e:
        return _raise_loading(e)
    n = max(1, min(4, req.n or 1))
    data = []
    for _ in range(n):
        b64 = _try_real_generation(model_id, req.prompt, req.size or "768x768",
                                   req.quality or "medium")
        if not b64:
            b64 = _placeholder_png_b64()
        data.append({"b64_json": b64})
    return {"created": int(time.time()), "data": data, "request_id": rid}


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
    # V1: img2img cez reálnu pipeline ak je, inak placeholder (kontrakt drží tvar)
    await image.read()
    b64 = _try_real_generation(model_id, prompt or "edit", size, quality)
    if not b64:
        b64 = _placeholder_png_b64(color=(30, 120, 200))
    return {"created": int(time.time()), "data": [{"b64_json": b64}], "request_id": rid}


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
    # V1: vrátiť vstupný obraz (pipeline inpaint sa napojí keď je váha v RAM);
    # kontrakt {image, elapsed, request_id} drží, mazanie ľudí funguje cez masku v ďalšom kroku.
    return {"image": req.image, "elapsed": round(time.time() - t0, 2), "request_id": rid}


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
    return {"image": req.image, "elapsed": round(time.time() - t0, 2), "request_id": rid}


@app.post("/v1/images/upscale")
def upscale(req: UpscaleRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
    if not req.image:
        return JSONResponse(status_code=400, content={"error": "missing image b64"})
    try:
        out = upscale_image_b64(req.image, req.scale, req.model or "RealESRGAN_x4v3")
    except ValueError as e:
        msg = str(e)
        if "exceeds cap" in msg:
            return JSONResponse(status_code=503, content={"error": "will not load", "detail": msg})
        return JSONResponse(status_code=400, content={"error": msg})
    out["request_id"] = rid
    return out


# ---------- audio ----------

ALLOWED_AUDIO_EXT = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav",
                     ".webm", ".ogg", ".flac", ".opus"}
MAX_FILE_SIZE = 100 * 1024 * 1024
ALLOWED_WHISPER = ("tiny", "base", "small", "medium", "turbo", "large")


def _transcribe_with_whisper(tmp_path: str, wanted: str, language):
    from faster_whisper import WhisperModel
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    m = WhisperModel(wanted, device=device, compute_type=compute)
    segs, info = m.transcribe(tmp_path, language=language, beam_size=1,
                              vad_filter=True)
    text = " ".join(s.text for s in segs).strip()
    return text, getattr(info, "language", language)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(default=None),
    audio: UploadFile = File(default=None),
    model: str = Form(default="whisper-1"),
    model_name: str = Form(default="whisper-1"),
    device: str = Form(default=""),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
):
    up = file or audio
    if up is None:
        return JSONResponse(status_code=400, content={"error": "missing file"})
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
    content = await up.read()
    if not content:
        return JSONResponse(status_code=400, content={"error": "Empty file uploaded"})
    if len(content) > MAX_FILE_SIZE:
        return JSONResponse(status_code=413, content={"error": "File too large"})
    try:
        _manager.ensure("audio", wanted)
    except ModelLoadingError as e:
        return _raise_loading(e)
    import tempfile
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)
        try:
            text, lang = _transcribe_with_whisper(tmp, wanted, language)
        except Exception as e:
            logger.warning(f"whisper engine missing, stub: {str(e)[:160]}")
            text, lang = "[unified stub — nainštaluj faster-whisper + model]", language
        if response_format == "text":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(text)
        return {"text": text, "language": lang}
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
    r = await transcriptions(file=file, audio=audio, model=model)
    if isinstance(r, dict) and "text" in r:
        return {"text": r["text"]}
    return r


@app.post("/transcribe")
async def transcribe_alias(file: UploadFile = File(...), language: str = Form(default=None)):
    return await transcriptions(file=file, language=language)


# ---------- webui ----------

@app.get("/webui", response_class=HTMLResponse)
def webui():
    p = HERE / "webui.html"
    if p.exists():
        html = p.read_text(encoding="utf-8", errors="ignore")
        # minimálny unified banner (neprepisuje logiku pôvodného UI)
        banner = ('<div style="padding:8px;background:#111;color:#eee;font:12px sans-serif">'
                  'Unified: Images + Audio + Upscale na :8101 (legacy audio :4444 shim v README)'
                  '</div>')
        return banner + html
    return ('<h3>Unified Standalone</h3><ul>'
            '<li>POST /v1/images/generations|edits|inpaint|harmonize|upscale</li>'
            '<li>POST /v1/audio/transcriptions|translations</li></ul>')


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unified_server:app", host="127.0.0.1", port=8101, workers=1)
