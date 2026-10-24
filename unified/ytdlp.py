"""yt-dlp wrapper for audio URLs (YouTube / direct links)."""
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("unified.ytdlp")


class YtDlpMissing(Exception):
    pass


class DownloadFailed(Exception):
    pass


def _have_ytdlp() -> bool:
    try:
        import yt_dlp  # noqa: F401
        return True
    except Exception:
        return shutil.which("yt-dlp") is not None


def _run(url: str, tmpdir: str) -> str:
    out_tmpl = str(Path(tmpdir) / "yt_%(id)s.%(ext)s")
    cmd = ["yt-dlp", "-x", "--audio-format", "wav", "--audio-quality", "0",
           "--postprocessor-args", "-ar 16000 -ac 1", "-o", out_tmpl, url]
    # Prefer the Python module if the binary is missing
    if shutil.which("yt-dlp") is None:
        cmd = ["python", "-m", "yt_dlp"] + cmd[1:]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise DownloadFailed(r.stderr[-400:] or r.stdout[-400:])
    # find the produced wav
    for p in sorted(Path(tmpdir).glob("yt_*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.stat().st_size > 0:
            return str(p)
    raise DownloadFailed("yt-dlp did not produce a wav output")


def download_audio(url: str, tmpdir: str | None = None, timeout: int = 600) -> str:
    if not url or not isinstance(url, str) or "://" not in url:
        raise DownloadFailed(f"invalid URL: {url!r}")
    if not _have_ytdlp():
        raise YtDlpMissing("pip install yt-dlp or the yt-dlp binary (https://github.com/yt-dlp/yt-dlp)")
    td = tmpdir or tempfile.mkdtemp(prefix="ytdlp_")
    Path(td).mkdir(parents=True, exist_ok=True)
    try:
        return _run(url, td)
    except subprocess.TimeoutExpired:
        raise DownloadFailed("yt-dlp timeout")
    except Exception as e:
        if isinstance(e, (YtDlpMissing, DownloadFailed)):
            raise
        raise DownloadFailed(str(e)[:400])
