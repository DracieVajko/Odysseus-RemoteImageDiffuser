"""LCM-Dreamshaper v7 INT8 ONNX pipeline (PC reference for the Android port).

Same model the Android app downloads (latentdivergence/lcm-dreamshaper-int8-onnx).
Goal: verify the exact ONNX signatures and the LCM procedure in Python, then port
1:1 to Kotlin (without guessing).

Usage:
    from unified.lcm_onnx import LcmOnnxPipeline
    p = LcmOnnxPipeline(models_dir)
    img = p.generate("a red cube", steps=4, seed=1)
"""
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("unified.lcm_onnx")

VAE_SCALE = 0.18215
DEFAULT_SIZE = 512
DEFAULT_STEPS = 4
DEFAULT_ORIGINAL_STEPS = 50
DEFAULT_GUIDANCE = 1.0  # LCM: no CFG (guidance <= 1)


class LcmOnnxPipeline:
    def __init__(self, model_dir: Path, threads: int = 0):
        import onnxruntime as ort
        self.dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        if threads > 0:
            opts.intra_op_num_threads = threads
        provs = ["CPUExecutionProvider"]

        def load(rel: str):
            p = self.dir / rel
            if not p.is_file():
                raise FileNotFoundError(p)
            t0 = time.time()
            sess = ort.InferenceSession(str(p), sess_options=opts, providers=provs)
            logger.info("loaded %s in %.1fs inputs=%s outputs=%s",
                        rel, time.time() - t0,
                        [i.name for i in sess.get_inputs()],
                        [o.name for o in sess.get_outputs()])
            return sess

        self.unet = load("unet/model.onnx")
        self.text_encoder = load("text_encoder/model.onnx")
        self.vae = load("vae_decoder/model.onnx")

        # Tokenizer z repo (CLIP BPE)
        from transformers import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(str(self.dir / "tokenizer"))

        # LCM scheduler z diffusers
        from diffusers import LCMScheduler
        self.scheduler = LCMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
            prediction_type="epsilon",
            timestep_spacing="leading",
            original_inference_steps=DEFAULT_ORIGINAL_STEPS,
        )

    # ---------- helpers ----------

    def _encode_prompt(self, prompt: str) -> np.ndarray:
        tok = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="np",
        )
        names = [i.name for i in self.text_encoder.get_inputs()]
        id_type = next(i.type for i in self.text_encoder.get_inputs() if i.name == "input_ids")
        input_ids = tok["input_ids"].astype(np.int32 if "int32" in id_type else np.int64)
        inputs = {}
        inputs["input_ids"] = input_ids
        if "attention_mask" in names:
            inputs["attention_mask"] = tok["attention_mask"].astype(np.int64)
        out = self.text_encoder.run(None, inputs)
        # obvykle last_hidden_state
        return out[0].astype(np.float32)

    @staticmethod
    def guidance_embedding(w: float, dim: int = 256) -> np.ndarray:
        """diffusers get_guidance_scale_embedding (LCM timestep_cond)."""
        wv = w * 1000.0
        half = dim // 2
        emb = np.log(10000.0) / (half - 1)
        emb = np.exp(np.arange(half, dtype=np.float32) * -emb)
        emb = wv * emb[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
        if dim % 2 == 1:
            emb = np.pad(emb, ((0, 0), (0, 1)))
        return emb.astype(np.float32)

    def _run_unet(self, sample, timestep, embeds, guidance: float = 1.0) -> np.ndarray:
        names = [i.name for i in self.unet.get_inputs()]
        feed = {}
        feed["sample" if "sample" in names else names[0]] = sample.astype(np.float32)
        tname = "timestep" if "timestep" in names else names[1]
        tinfo = next(i for i in self.unet.get_inputs() if i.name == tname)
        if tinfo.type == "tensor(int64)":
            feed[tname] = np.array([int(timestep)], dtype=np.int64)
        elif tinfo.type == "tensor(float)":
            feed[tname] = np.array([float(timestep)], dtype=np.float32)
        else:
            feed[tname] = np.array(int(timestep), dtype=np.int64)
        ename = "encoder_hidden_states" if "encoder_hidden_states" in names else names[2]
        feed[ename] = embeds.astype(np.float32)
        if "timestep_cond" in names:
            feed["timestep_cond"] = self.guidance_embedding(max(0.0, guidance - 1.0))
        return self.unet.run(None, feed)[0]

    def _decode_latents(self, latents: np.ndarray) -> np.ndarray:
        names = [i.name for i in self.vae.get_inputs()]
        feed = {("latent_sample" if "latent_sample" in names else names[0]):
                (latents / VAE_SCALE).astype(np.float32)}
        out = self.vae.run(None, feed)[0]
        img = (out / 2 + 0.5).clip(0, 1)
        # (1,3,H,W) -> (H,W,3) uint8
        img = (img[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
        return img

    # ---------- main ----------

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = DEFAULT_STEPS,
        size: int = DEFAULT_SIZE,
        seed: int | None = None,
        guidance: float = DEFAULT_GUIDANCE,
    ):
        from PIL import Image

        t_start = time.time()
        embeds = self._encode_prompt(prompt)
        logger.info("prompt embedded in %.1fs shape=%s", time.time() - t_start, embeds.shape)

        h = w = size // 8
        rng = np.random.default_rng(seed)
        latents = rng.standard_normal((1, 4, h, w)).astype(np.float32)

        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps
        latents = latents * self.scheduler.init_noise_sigma
        logger.info("steps=%s timesteps=%s", steps, timesteps.tolist())

        t_loop = time.time()
        for i, t in enumerate(timesteps):
            model_input = self.scheduler.scale_model_input(latents, t)
            noise = self._run_unet(model_input, t, embeds, guidance)
            import torch
        latents_t = torch.from_numpy(latents)
        noise_t = torch.from_numpy(noise)
        t_tensor = torch.tensor([t], dtype=torch.long)
        step_out = self.scheduler.step(noise_t, t_tensor, latents_t)
        latents = step_out.prev_sample.detach().cpu().numpy()
        logger.info("step %d/%d t=%s %.1fs", i + 1, len(timesteps), int(t),
                        time.time() - t_loop)

        arr = self._decode_latents(latents)
        logger.info("total %.1fs", time.time() - t_start)
        return Image.fromarray(arr)


def find_lcm_dir(base: Path | None = None) -> Path | None:
    base = base or Path(__file__).resolve().parent.parent / "models"
    d = base / "LCM_Dreamshaper_ONNX"
    if (d / "unet" / "model.onnx").is_file():
        return d
    return None
