"""Download NVIDIA Nemotron 3.5 ASR (0.6B) for the Prism PC server.

Usage:
    python pc/tools/download_nemotron_asr.py

Fetches the bfloat16 safetensors checkpoint (~2.55 GB) plus the processor and
tokenizer files into pc/models/nemotron-asr-0.6b/.

The optional GGUF (0.74 GB) is left alone: llama.cpp / NeMo-Speech.cpp support
for this architecture is not wired into Prism yet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "nemotron-asr-0.6b"

FILES = [
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    missing = [f for f in FILES if not (MODEL_DIR / f).exists()]
    if not missing:
        print(f"Already installed: {MODEL_DIR}")
        return
    print(f"Downloading {MODEL_ID} -> {MODEL_DIR}")
    print(f"  missing: {', '.join(missing)}")
    from huggingface_hub import hf_hub_download

    for name in missing:
        print(f"  {name} ...", flush=True)
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=name,
            local_dir=str(MODEL_DIR),
            revision="main",
        )
    print("Done.")
    for p in sorted(MODEL_DIR.iterdir()):
        if p.is_file():
            print(f"  {p.name}: {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
