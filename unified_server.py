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
import tempfile
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from typing import Optional, Tuple

from unified.audio_proxy import transcribe_via_service
from unified.audio_splitter import transcribe_with_splitting, get_audio_duration
from unified.ffmpeg_tools import (
    VIDEO_PRESETS, AUDIO_PRESETS, RESIZE_PRESETS,
    convert_video, convert_audio, extract_audio, trim_media, compress_video,
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


@asynccontextmanager
async def lifespan(application):
    # Best effort: registry bez váh; reálna inferencia až keď sú torch+váhy.
    try:
        if os.getenv("UNIFIED_NO_ENGINE", "") != "1":
            init_diffusion_engine()
    except Exception as e:
        logger.warning(f"engine init skipped: {e}")
    yield


app = FastAPI(title="Unified Standalone Server", lifespan=lifespan)
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
    _release_vlm_best_effort()
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
    _release_vlm_best_effort()
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
    _release_vlm_best_effort()
    # Reálna pipeline keď je engine naložený (masks = mazanie ľudí v pozadí),
    # inak echo vstupu — kontrakt {image, elapsed, request_id} drží vždy.
    if req.mask:
        try:
            real = try_real_inpaint(req.image, req.mask, req.prompt, model_id,
                                    req.strength, req.steps)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)[:200]})
        except Exception as e:
            logger.warning(f"real inpaint failed, echo fallback: {str(e)[:160]}")
            real = None
        if real:
            return {"image": real, "elapsed": round(time.time() - t0, 2),
                    "request_id": rid, "engine": "diffusion"}
    return {"image": req.image, "elapsed": round(time.time() - t0, 2),
            "request_id": rid, "engine": "echo"}


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


# ---------- vision (V2) ----------

@app.post("/v1/images/caption")
def caption(req: CaptionRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:8]}"
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
    return out


def _chat_vision_parts(messages):
    """Z OpenAI messages vytiahni (text, image_b64) poslednej user správy."""
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
    try:
        _manager.ensure("vlm", VLM_ID)
    except ModelLoadingError as e:
        return _raise_loading(e)
    text, image = _chat_vision_parts(messages)
    if not image:
        return JSONResponse(status_code=400, content={
            "error": "missing image: V1 podporuje len vision správy (content s image_url data:base64)"})
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


async def _transcribe_with_splitting(audio_path: Path, wanted: str, language):
    """Transcribe with auto-splitting for long audio files."""
    def _transcribe_single(path: Path, lang: Optional[str]) -> Tuple[str, str]:
        text, lang = _transcribe_with_whisper(str(path), wanted, lang)
        return text, lang
    
    try:
        # Use the audio splitter utility
        from unified.audio_splitter import transcribe_with_splitting
        text, lang = await transcribe_with_splitting(audio_path, _transcribe_single, language)
        return text, lang
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
):
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
    import tempfile
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
            tmp = t.name
            t.write(content)
        try:
            split_enabled = auto_split.lower() in ("true", "1", "yes", "on")
            if split_enabled:
                text, lang = await _transcribe_with_splitting(Path(tmp), wanted, language)
            else:
                text, lang = _transcribe_with_whisper(tmp, wanted, language)
            engine = "local"
        except Exception as e:
            logger.warning("local whisper missing (%s), proxy %s",
                             str(e)[:120], WHISPER_SERVICE_URL)
            proxied = transcribe_via_service(
                WHISPER_SERVICE_URL, content, up.filename or "audio.wav",
                wanted, language)
            if proxied is not None:
                text, lang = proxied
                engine = "whisper-service"
                # nič lokálne nedržíme v RAM → gate hneď uvoľniť pre obrazy
                _manager.unload("audio-proxied")
            else:
                text, lang = "[unified stub — spusti WhisperApi službu alebo nainštaluj faster-whisper]", language
                engine = "stub"
        if response_format == "text":
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(text)
        return {"text": text, "language": lang, "engine": engine}
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

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
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

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
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

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
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

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
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

        try:
            Path(out_path).unlink(missing_ok=True)
        except Exception:
            pass

        return {
            "file": out_b64,
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




# ---------- model install/uninstall (PC) ----------

from unified import models_manager as _mm


class InstallRequest(BaseModel):
    id: str = ""


@app.get("/v1/models/catalog")
def models_catalog():
    """Katalóg inštalovateľných modelov + stav (installed/downloading/…)."""
    try:
        return {"models": _mm.catalog()}
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




@app.get("/v1/settings")
def settings_info():
    """Runtime info pre WebUI Settings tab (read-only, žiadne tajomstvá)."""
    import shutil as _sh
    torch_ok = False
    cuda = False
    try:
        import torch
        torch_ok = True
        cuda = torch.cuda.is_available()
    except Exception:
        pass
    return {
        "version": "unified-0.1.0",
        "port": 8101,
        "whisper_url": WHISPER_SERVICE_URL,
        "models_dir": str(HERE / "models"),
        "output_dir": str(HERE / "output"),
        "default_image_model": _image_registry().get("default_image_model", ""),
        "audio_models": list(ALLOWED_WHISPER),
        "engines": {
            "torch": torch_ok,
            "cuda": cuda,
            "ffmpeg": _sh.which("ffmpeg") or "",
            "ffprobe": _sh.which("ffprobe") or "",
            "ytdlp": _sh.which("yt-dlp") or "",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unified_server:app", host="127.0.0.1", port=8101, workers=1)
