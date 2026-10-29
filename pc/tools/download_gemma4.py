"""Download Gemma4 E2B model for Prism PC.

Usage:
    python pc/tools/download_gemma4.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "gemma-4-E2B-it"

def main():
    if (MODEL_DIR / "model.safetensors").exists():
        print(f"Already installed: {MODEL_DIR}")
        return
    print(f"Downloading {MODEL_ID} to {MODEL_DIR}...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"Done. Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
