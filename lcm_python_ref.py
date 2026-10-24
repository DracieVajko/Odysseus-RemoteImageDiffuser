"""Standalone LCM-Dreamshaper ONNX pipeline (numpy + onnxruntime) for validation."""
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

BASE = Path(r"D:\AI\DiffusionServer\models\LCM_Dreamshaper_ONNX")
VAE_SCALE = 0.18215
NUM_TRAIN = 1000
ORIGINAL_STEPS = 50
SIGMA_DATA = 0.5
TIMESTEP_SCALING = 10.0
MAX_LEN = 77
BOS, EOS = 49406, 49407

vocab = json.loads((BASE / "tokenizer" / "vocab.json").read_text(encoding="utf-8"))
merges = []
for line in (BASE / "tokenizer" / "merges.txt").read_text(encoding="utf-8").splitlines()[1:]:
    parts = line.strip().split(" ")
    if len(parts) == 2:
        merges.append((parts[0], parts[1]))
rank = {p: i for i, p in enumerate(merges)}


def bpe(word):
    symbols = list(word)
    if not symbols:
        return []
    symbols[-1] += "</w>"
    while len(symbols) > 1:
        best = None
        best_i = -1
        for i in range(len(symbols) - 1):
            r = rank.get((symbols[i], symbols[i + 1]))
            if r is not None and (best is None or r < best):
                best = r
                best_i = i
        if best_i < 0:
            break
        symbols[best_i] += symbols[best_i + 1]
        del symbols[best_i + 1]
    return symbols


def tokenize(text):
    tokens = ["<|startoftext|>"]
    for w in text.lower().strip().split():
        tokens.extend(bpe(w))
    tokens.append("<|endoftext|>")
    ids = [vocab.get(t, EOS) for t in tokens][:MAX_LEN]
    ids += [EOS] * (MAX_LEN - len(ids))
    return ids


# scheduler tables
betas = np.array([(math.sqrt(0.00085) + (math.sqrt(0.012) - math.sqrt(0.00085)) * i / (NUM_TRAIN - 1)) ** 2
                  for i in range(NUM_TRAIN)], dtype=np.float32)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)
final_alpha_cumprod = float(alphas_cumprod[0])


def timesteps_for(steps):
    k = NUM_TRAIN // ORIGINAL_STEPS
    origin = [(i + 1) * k - 1 for i in range(ORIGINAL_STEPS)]
    rev = origin[::-1]
    idx = [math.floor(i * ORIGINAL_STEPS / steps) for i in range(steps)]
    return [rev[i] for i in idx]


def guidance_embedding(w, dim=256):
    wv = w * 1000.0
    half = dim // 2
    emb = np.exp(-math.log(10000.0) / (half - 1) * np.arange(half, dtype=np.float32))
    emb = wv * emb[None, :]
    return np.concatenate([np.sin(emb), np.cos(emb)], axis=1).astype(np.float32)


providers = ["CPUExecutionProvider"]
te = ort.InferenceSession(str(BASE / "text_encoder" / "model.onnx"), providers=providers)
unet = ort.InferenceSession(str(BASE / "unet" / "model_qdq_inline.onnx"), providers=providers)
vae = ort.InferenceSession(str(BASE / "vae_decoder" / "model.onnx"), providers=providers)


def generate(prompt, size=512, steps=4, seed=1):
    ids = np.array([tokenize(prompt)], dtype=np.int32)
    emb = te.run(None, {"input_ids": ids})[0].astype(np.float32)
    print("emb", emb.shape)

    h = w = size // 8
    rng = np.random.default_rng(seed)
    latents = rng.standard_normal((1, 4, h, w)).astype(np.float32)
    ts = timesteps_for(steps)
    print("timesteps", ts)
    unet_inputs = [i.name for i in unet.get_inputs()]
    for step, t in enumerate(ts):
        feed = {"sample": latents, "timestep": np.array([t], dtype=np.int64),
                "encoder_hidden_states": emb}
        if "timestep_cond" in unet_inputs:
            feed["timestep_cond"] = guidance_embedding(0.0)
        noise = unet.run(None, feed)[0].astype(np.float32)
        # scheduler step
        a_t = float(alphas_cumprod[t])
        prev_t = ts[step + 1] if step + 1 < len(ts) else t
        a_prev = float(alphas_cumprod[prev_t]) if prev_t >= 0 else final_alpha_cumprod
        b_t, b_prev = 1 - a_t, 1 - a_prev
        scaled = t * TIMESTEP_SCALING
        c_skip = SIGMA_DATA ** 2 / (scaled ** 2 + SIGMA_DATA ** 2)
        c_out = scaled / math.sqrt(scaled ** 2 + SIGMA_DATA ** 2)
        x0 = (latents - math.sqrt(b_t) * noise) / math.sqrt(a_t)
        denoised = c_out * x0 + c_skip * latents
        if step != len(ts) - 1:
            z = rng.standard_normal(latents.shape).astype(np.float32)
            latents = math.sqrt(a_prev) * denoised + math.sqrt(b_prev) * z
        else:
            latents = denoised
        print(f"step {step+1}/{len(ts)} t={t}")

    img = vae.run(None, {"latent_sample": latents / VAE_SCALE})[0]
    img = (img / 2 + 0.5).clip(0, 1)
    img = (img[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return img


if __name__ == "__main__":
    for size in (256, 512):
        arr = generate("erruption volcano", size=size, steps=4)
        out = Path(rf"D:\AI\DiffusionServer\py_lcm_{size}.png")
        # save PNG via PIL-free minimal writer
        from struct import pack
        import zlib
        h, w, _ = arr.shape
        raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))
        def chunk(t, d):
            c = pack(">I", len(d)) + t + d
            return c + pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
        out.write_bytes(png)
        print("saved", out, "bytes", len(png))
