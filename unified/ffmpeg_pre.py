"""Audio/video normalizácia — ffmpeg ak je, inak passthrough wav."""
import shutil
import subprocess
from pathlib import Path

ALLOWED_AUDIO_EXT = {
    ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a",
    ".wav", ".webm", ".ogg", ".flac", ".opus",
}
MAX_FILE_SIZE_MB = 100


class UnsupportedFormat(Exception):
    pass


class EmptyFile(Exception):
    pass


class TooLarge(Exception):
    pass


def normalize_audio_to_wav16k(src: Path, dst: Path) -> dict:
    src = Path(src)
    dst = Path(dst)
    ext = src.suffix.lower()
    if ext not in ALLOWED_AUDIO_EXT:
        raise UnsupportedFormat(f"Unsupported audio format: {ext}")
    if not src.exists() or src.stat().st_size == 0:
        raise EmptyFile("Empty file")
    if src.stat().st_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise TooLarge("File too large")
    ff = shutil.which("ffmpeg")
    if ff and ext != ".wav":
        dst.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [ff, "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
            capture_output=True, timeout=120,
        )
        if r.returncode == 0 and dst.exists():
            return {"path": str(dst), "via": "ffmpeg"}
    # passthrough (wav alebo bez ffmpeg): skopíruj
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    return {"path": str(dst), "via": "passthrough"}
