"""LCM-Dreamshaper ONNX pipeline (numpy + onnxruntime), no torch/diffusers.

Supports txt2img with optional classifier-free guidance (negative prompt),
img2img and inpaint (mask blend), plus a non-AI color-replace mode.

Used by the PC server when torch/diffusers are unavailable, and as the
reference for the Android port.
"""
import base64
import io
import json
import logging
import math
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger("unified.lcm_onnx_numpy")

VAE_SCALE = 0.18215
NUM_TRAIN = 1000
ORIGINAL_STEPS = 50
SIGMA_DATA = 0.5
TIMESTEP_SCALING = 10.0
MAX_LEN = 77
BOS, EOS = 49406, 49407


def find_lcm_dir(base: Path | None = None) -> Path | None:
    base = base or Path(__file__).resolve().parent.parent / "models"
    for name in ("LCM_Dreamshaper_ONNX", "lcm-dreamshaper-int8-onnx"):
        d = base / name
        if (d / "unet" / "model.onnx").is_file() or (d / "unet" / "model_qdq_inline.onnx").is_file():
            return d
    return None


def _b64_image_to_pil(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _to_nchw(img: Image.Image, size: int) -> np.ndarray:
    img = img.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    return arr.transpose(2, 0, 1)[None, ...]


def _to_pil(arr: np.ndarray) -> Image.Image:
    arr = (arr[0].transpose(1, 2, 0) * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


class LcmOnnxNumpy:
    def __init__(self, model_dir: Path, use_gpu: bool | None = None):
        import os
        import onnxruntime as ort
        from unified.onnx_env import pick_providers
        self.dir = Path(model_dir)
        self._vocab = None
        self._rank = None
        self._load_tokenizer()
        self._build_scheduler()
        # Provider selection:
        #   None  -> auto (CUDA if available, DML skipped for QDQ, else CPU)
        #   True  -> force CUDA when available
        #   False -> force CPU
        if use_gpu is None:
            provs = pick_providers("qdq")
        elif use_gpu:
            avail = []
            try:
                avail = ort.get_available_providers()
            except Exception:
                pass
            provs = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
        else:
            provs = ["CPUExecutionProvider"]
        logger.info("ONNX providers: %s", provs)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        if provs != ["CPUExecutionProvider"]:
            # Without this, ORT pre-allocates a memory pattern per input shape;
            # on a 6 GB card mixed 256/384/512 requests then spill to shared
            # memory (measured 213 s for one 384 request vs 17 s with it off).
            so.enable_mem_pattern = False
        if provs == ["CPUExecutionProvider"]:
            try:
                so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
            except Exception:
                pass
        unet = self.dir / "unet" / "model_qdq_inline.onnx"
        if not unet.is_file():
            unet = self.dir / "unet" / "model.onnx"
        self.unet = ort.InferenceSession(str(unet), sess_options=so, providers=provs)
        self.te = ort.InferenceSession(str(self.dir / "text_encoder" / "model.onnx"), sess_options=so, providers=provs)
        self.vae = ort.InferenceSession(str(self.dir / "vae_decoder" / "model.onnx"), sess_options=so, providers=provs)
        vae_enc = self.dir / "vae_encoder" / "model.onnx"
        self.vae_enc = ort.InferenceSession(str(vae_enc), sess_options=so, providers=provs) if vae_enc.is_file() else None
        self.unet_inputs = [i.name for i in self.unet.get_inputs()]

    # ---------- tokenizer ----------

    def _load_tokenizer(self):
        base = self.dir / "tokenizer"
        self._vocab = json.loads((base / "vocab.json").read_text(encoding="utf-8"))
        merges = []
        for line in (base / "merges.txt").read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.strip().split(" ")
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
        self._rank = {p: i for i, p in enumerate(merges)}

    def _bpe(self, word: str):
        symbols = list(word)
        if not symbols:
            return []
        symbols[-1] += "</w>"
        while len(symbols) > 1:
            best = None
            best_i = -1
            for i in range(len(symbols) - 1):
                r = self._rank.get((symbols[i], symbols[i + 1]))
                if r is not None and (best is None or r < best):
                    best = r
                    best_i = i
            if best_i < 0:
                break
            symbols[best_i] += symbols[best_i + 1]
            del symbols[best_i + 1]
        return symbols

    def _tokenize(self, text: str):
        tokens = ["<|startoftext|>"]
        for w in (text or "").lower().strip().split():
            tokens.extend(self._bpe(w))
        tokens.append("<|endoftext|>")
        ids = [self._vocab.get(t, EOS) for t in tokens][:MAX_LEN]
        ids += [EOS] * (MAX_LEN - len(ids))
        return ids

    # ---------- scheduler ----------

    def _build_scheduler(self):
        betas = np.array([
            (math.sqrt(0.00085) + (math.sqrt(0.012) - math.sqrt(0.00085)) * i / (NUM_TRAIN - 1)) ** 2
            for i in range(NUM_TRAIN)
        ], dtype=np.float32)
        self.alphas_cumprod = np.cumprod(1.0 - betas)
        self.final_alpha_cumprod = float(self.alphas_cumprod[0])

    def _timesteps(self, steps: int):
        k = NUM_TRAIN // ORIGINAL_STEPS
        origin = [(i + 1) * k - 1 for i in range(ORIGINAL_STEPS)]
        rev = origin[::-1]
        idx = [math.floor(i * ORIGINAL_STEPS / steps) for i in range(steps)]
        return [rev[i] for i in idx]

    @staticmethod
    def _guidance_embedding(w: float, dim: int = 256):
        wv = w * 1000.0
        half = dim // 2
        emb = np.exp(-math.log(10000.0) / (half - 1) * np.arange(half, dtype=np.float32))
        emb = wv * emb[None, :]
        return np.concatenate([np.sin(emb), np.cos(emb)], axis=1).astype(np.float32)

    @staticmethod
    def _noise_like(shape, seed: int):
        return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)

    # ---------- inference ----------

    def _encode_text(self, text: str) -> np.ndarray:
        ids = np.array([self._tokenize(text)], dtype=np.int32)
        return self.te.run(None, {"input_ids": ids})[0].astype(np.float32)

    def _run_unet(self, latents, t, emb, guidance):
        feed = {
            "sample": latents,
            "timestep": np.array([t], dtype=np.int64),
            "encoder_hidden_states": emb,
        }
        if "timestep_cond" in self.unet_inputs:
            feed["timestep_cond"] = self._guidance_embedding(max(0.0, guidance - 1.0))
        return self.unet.run(None, feed)[0].astype(np.float32)

    def _step(self, latents, noise, t, step, ts):
        a_t = float(self.alphas_cumprod[t])
        prev_t = ts[step + 1] if step + 1 < len(ts) else t
        a_prev = float(self.alphas_cumprod[prev_t]) if prev_t >= 0 else self.final_alpha_cumprod
        b_t, b_prev = 1 - a_t, 1 - a_prev
        scaled = t * TIMESTEP_SCALING
        c_skip = SIGMA_DATA ** 2 / (scaled ** 2 + SIGMA_DATA ** 2)
        c_out = scaled / math.sqrt(scaled ** 2 + SIGMA_DATA ** 2)
        x0 = (latents - math.sqrt(b_t) * noise) / math.sqrt(a_t)
        denoised = c_out * x0 + c_skip * latents
        if step != len(ts) - 1:
            z = self._noise_like(latents.shape, 1000 + step)
            return math.sqrt(a_prev) * denoised + math.sqrt(b_prev) * z
        return denoised

    def _denoise(self, latents, pos_emb, neg_emb, guidance, steps, mask=None,
                 original=None, timesteps=None):
        # Use the caller's schedule when provided (img2img/inpaint slice it);
        # otherwise build the standard LCM schedule for `steps`.
        ts = list(timesteps) if timesteps is not None else self._timesteps(max(1, min(8, steps)))
        for step, t in enumerate(ts):
            noise_pos = self._run_unet(latents, t, pos_emb, guidance)
            if guidance > 1.0 and neg_emb is not None:
                noise_neg = self._run_unet(latents, t, neg_emb, guidance)
                noise = noise_neg + guidance * (noise_pos - noise_neg)
            else:
                noise = noise_pos
            latents = self._step(latents, noise, t, step, ts)
            if mask is not None and original is not None:
                latents = mask * latents + (1.0 - mask) * original
        return latents

    def _decode(self, latents) -> Image.Image:
        out = self.vae.run(None, {"latent_sample": latents / VAE_SCALE})[0]
        out = (out / 2 + 0.5).clip(0, 1)
        return _to_pil(out)

    def _encode_latents(self, img: Image.Image, size: int):
        if self.vae_enc is None:
            raise RuntimeError("vae_encoder/model.onnx is missing")
        x = _to_nchw(img, size)
        lat = self.vae_enc.run(None, {"sample": x})[0].astype(np.float32)
        return lat * VAE_SCALE

    # ---------- public ----------

    def generate(self, prompt: str, negative_prompt: str = "", guidance: float = 1.0,
                 size: int = 512, steps: int = 6, seed: int = 1) -> Image.Image:
        pos = self._encode_text(prompt)
        neg = self._encode_text(negative_prompt) if (negative_prompt and guidance > 1.0) else None
        h = w = size // 8
        latents = self._noise_like((1, 4, h, w), seed)
        latents = self._denoise(latents, pos, neg, guidance, steps)
        return self._decode(latents)

    def img2img(self, image_b64: str, prompt: str, negative_prompt: str = "",
                guidance: float = 1.0, strength: float = 0.75, size: int = 512,
                steps: int = 6, seed: int = 1) -> Image.Image:
        src = _b64_image_to_pil(image_b64)
        latents = self._encode_latents(src, size)
        pos = self._encode_text(prompt)
        neg = self._encode_text(negative_prompt) if (negative_prompt and guidance > 1.0) else None
        # start at a later timestep based on strength
        start = int(round((1.0 - max(0.05, min(1.0, strength))) * steps))
        ts = self._timesteps(max(1, min(8, steps)))[start:]
        t0 = ts[0]
        a0 = float(self.alphas_cumprod[t0])
        noise = self._noise_like(latents.shape, seed)
        latents = math.sqrt(a0) * latents + math.sqrt(1 - a0) * noise
        latents = self._denoise(latents, pos, neg, guidance, len(ts), timesteps=ts)
        return self._decode(latents)

    def inpaint(self, image_b64: str, mask_b64: str, prompt: str, negative_prompt: str = "",
                guidance: float = 1.0, strength: float = 0.8, size: int = 512,
                steps: int = 6, seed: int = 1, mode: str = "add",
                color: str = "#0066ff") -> Image.Image:
        src = _b64_image_to_pil(image_b64)
        mask_img = _b64_image_to_pil(mask_b64).convert("L").resize((size, size), Image.NEAREST)
        if mode == "color":
            return _colorize(src, mask_img, color)

        latents = self._encode_latents(src, size)
        h = w = size // 8
        # latent mask (1,1,h,w), feathered
        m = mask_img.resize((w, h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=2))
        m = np.asarray(m).astype(np.float32) / 255.0
        m = m[None, None, ...]
        pos = self._encode_text(prompt)
        neg = self._encode_text(negative_prompt) if (negative_prompt and guidance > 1.0) else None

        start = int(round((1.0 - max(0.05, min(1.0, strength))) * steps))
        ts = self._timesteps(max(1, min(8, steps)))[start:]
        t0 = ts[0]
        a0 = float(self.alphas_cumprod[t0])
        noise = self._noise_like(latents.shape, seed)
        original = math.sqrt(a0) * latents + math.sqrt(1 - a0) * noise
        cur = original.copy()
        cur = self._denoise(cur, pos, neg, guidance, len(ts), mask=m, original=original, timesteps=ts)
        out = self._decode(cur).resize(src.size, Image.LANCZOS)
        # composite the feathered mask over the original
        m_full = mask_img.filter(ImageFilter.GaussianBlur(radius=3)).resize(src.size, Image.BILINEAR)
        return Image.composite(out, src, m_full)


def _colorize(src: Image.Image, mask: Image.Image, color: str) -> Image.Image:
    """Recolor the masked region to `color`, preserving luminance."""
    import colorsys
    hex_color = color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    tr, tg, tb = (int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    th, ts, _ = colorsys.rgb_to_hsv(tr, tg, tb)
    src_rgb = np.asarray(src.convert("RGB")).astype(np.float32) / 255.0
    hsv = np.asarray(src.convert("HSV")).astype(np.float32) / 255.0
    hsv[..., 0] = th
    hsv[..., 1] = np.clip(ts, 0.35, 1.0)
    out = Image.fromarray((hsv * 255).astype(np.uint8), "HSV").convert("RGB")
    if mask.size != src.size:
        mask = mask.resize(src.size, Image.BILINEAR)
    return Image.composite(out, src, mask)


def harmonize_b64(image_b64: str, mask_b64: str = "", color_match: bool = True,
                  seam_fix: bool = True, strength: float = 0.5) -> str:
    """Non-AI harmonize: match the masked region's color to the background and
    feather the seam. Without a mask, apply a mild global auto-level."""
    src = _b64_image_to_pil(image_b64)
    if not mask_b64:
        # mild global contrast/level normalization
        arr = np.asarray(src).astype(np.float32)
        lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
        if hi > lo:
            arr = (arr - lo) * (255.0 / (hi - lo))
        return _pil_to_b64(Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB"))

    mask = _b64_image_to_pil(mask_b64).convert("L").resize(src.size, Image.NEAREST)
    m = np.asarray(mask).astype(np.float32) / 255.0
    src_arr = np.asarray(src).astype(np.float32)
    if color_match:
        inside = m > 0.5
        outside = m <= 0.5
        if inside.any() and outside.any():
            mean_in = src_arr[inside].mean(axis=0)
            mean_out = src_arr[outside].mean(axis=0)
            shift = (mean_out - mean_in) * float(np.clip(strength, 0.0, 1.0))
            src_arr = src_arr + shift[None, None, :] * m[..., None]
    out = Image.fromarray(src_arr.clip(0, 255).astype(np.uint8), "RGB")
    if seam_fix:
        # blend a blurred version near the mask edge
        blurred = out.filter(ImageFilter.GaussianBlur(radius=2))
        soft = mask.filter(ImageFilter.GaussianBlur(radius=3))
        out = Image.composite(blurred, out, soft)
    return _pil_to_b64(out)


def _png_bytes(arr) -> bytes:
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


_PIPELINE = None


def _pipeline(model_dir: Path) -> LcmOnnxNumpy:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = LcmOnnxNumpy(model_dir)
    return _PIPELINE


def release_pipeline() -> bool:
    """Drop the cached pipeline so the GPU/CPU memory is actually freed."""
    global _PIPELINE
    had = _PIPELINE is not None
    _PIPELINE = None
    try:
        import gc
        gc.collect()
    except Exception:
        pass
    return had


def generate_b64(model_dir: Path, prompt: str, negative_prompt: str = "", guidance: float = 1.0,
                 size: int = 512, steps: int = 6, seed: int = 1) -> str:
    img = _pipeline(model_dir).generate(prompt, negative_prompt, guidance, size, steps, seed)
    return _pil_to_b64(img)


def img2img_b64(model_dir: Path, image_b64: str, prompt: str, negative_prompt: str = "",
                guidance: float = 1.0, strength: float = 0.75, size: int = 512,
                steps: int = 6, seed: int = 1) -> str:
    img = _pipeline(model_dir).img2img(image_b64, prompt, negative_prompt, guidance,
                                       strength, size, steps, seed)
    return _pil_to_b64(img)


def inpaint_b64(model_dir: Path, image_b64: str, mask_b64: str, prompt: str,
                negative_prompt: str = "", guidance: float = 1.0, strength: float = 0.8,
                size: int = 512, steps: int = 6, seed: int = 1,
                mode: str = "add", color: str = "#0066ff") -> str:
    img = _pipeline(model_dir).inpaint(image_b64, mask_b64, prompt, negative_prompt, guidance,
                                       strength, size, steps, seed, mode, color)
    return _pil_to_b64(img)
