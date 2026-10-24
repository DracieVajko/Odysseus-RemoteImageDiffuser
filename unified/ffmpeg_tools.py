"""FFmpeg tools — conversions, compression, extraction, trim, resize.

Presets for common tasks. Each function returns (output_path, info_dict).
"""
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Video conversion presets
VIDEO_PRESETS = {
    "mp4-h264": {
        "desc": "MP4 H.264 (universal, web/compatibility)",
        "vcodec": "libx264", "acodec": "aac",
        "crf": 23, "preset": "medium", "audio_bitrate": "128k",
        "ext": ".mp4",
    },
    "mp4-h265": {
        "desc": "MP4 H.265/HEVC (smaller files, newer devices)",
        "vcodec": "libx265", "acodec": "aac",
        "crf": 28, "preset": "medium", "audio_bitrate": "128k",
        "ext": ".mp4",
    },
    "mp4-small": {
        "desc": "MP4 small (CRF 32, slow, for sharing)",
        "vcodec": "libx264", "acodec": "aac",
        "crf": 32, "preset": "slow", "audio_bitrate": "96k",
        "ext": ".mp4",
    },
    "webm": {
        "desc": "WebM (web, VP9)",
        "vcodec": "libvpx-vp9", "acodec": "libopus",
        "crf": 32, "preset": None, "audio_bitrate": "128k",
        "ext": ".webm",
    },
    "mp4-h264-hq": {
        "desc": "MP4 H.264 high quality (CRF 18, slow)",
        "vcodec": "libx264", "acodec": "aac", "crf": 18,
        "preset": "slow", "audio_bitrate": "192k", "ext": ".mp4",
    },
    "mp4-h264-fast": {
        "desc": "MP4 H.264 fast (CRF 28, veryfast)",
        "vcodec": "libx264", "acodec": "aac", "crf": 28,
        "preset": "veryfast", "audio_bitrate": "128k", "ext": ".mp4",
    },
    "mp4-h265-hq": {
        "desc": "MP4 H.265 high quality (CRF 24)",
        "vcodec": "libx265", "acodec": "aac", "crf": 24,
        "preset": "medium", "audio_bitrate": "192k", "ext": ".mp4",
    },
    "mp4-h265-fast": {
        "desc": "MP4 H.265 fast (CRF 30)",
        "vcodec": "libx265", "acodec": "aac", "crf": 30,
        "preset": "fast", "audio_bitrate": "128k", "ext": ".mp4",
    },
    "mp4-discord": {
        "desc": "MP4 for Discord (<10 MB)",
        "vcodec": "libx264", "acodec": "aac", "crf": 30,
        "preset": "fast", "audio_bitrate": "96k", "ext": ".mp4",
        "max_size_mb": 10,
    },
    "gif": {
        "desc": "GIF (animation, max 15s, 480px)",
        "vcodec": "gif", "acodec": None,
        "crf": None, "preset": None, "audio_bitrate": None,
        "ext": ".gif", "max_duration": 15, "scale": 480,
    },
}

# Audio conversion presets
AUDIO_PRESETS = {
    "mp3-128": {
        "desc": "MP3 128 kbps (standard)",
        "acodec": "libmp3lame", "audio_bitrate": "128k", "ext": ".mp3",
    },
    "mp3-192": {
        "desc": "MP3 192 kbps (balanced)",
        "acodec": "libmp3lame", "audio_bitrate": "192k", "ext": ".mp3",
    },
    "mp3-320": {
        "desc": "MP3 320 kbps (high quality)",
        "acodec": "libmp3lame", "audio_bitrate": "320k", "ext": ".mp3",
    },
    "aac-128": {
        "desc": "AAC 128 kbps (Apple/Android compatible)",
        "acodec": "aac", "audio_bitrate": "128k", "ext": ".m4a",
    },
    "aac-192": {
        "desc": "AAC 192 kbps (high quality)",
        "acodec": "aac", "audio_bitrate": "192k", "ext": ".m4a",
    },
    "wav-16k": {
        "desc": "WAV 16 kHz mono (for Whisper)",
        "acodec": "pcm_s16le", "audio_bitrate": None,
        "ar": 16000, "ac": 1, "ext": ".wav",
    },
    "opus": {
        "desc": "Opus (small, high quality)",
        "acodec": "libopus", "audio_bitrate": "96k", "ext": ".opus",
    },
    "flac": {
        "desc": "FLAC (lossless)",
        "acodec": "flac", "audio_bitrate": None, "ext": ".flac",
    },
}

# Image presets (single-frame conversion)
IMAGE_PRESETS = {
    "jpg-95": {"desc": "JPG quality 95 (max)", "vcodec": "mjpeg", "qscale": 2, "ext": ".jpg"},
    "jpg-85": {"desc": "JPG quality 85 (smaller)", "vcodec": "mjpeg", "qscale": 4, "ext": ".jpg"},
    "png": {"desc": "PNG (lossless)", "vcodec": "png", "ext": ".png"},
    "webp": {"desc": "WebP (small, high quality)", "vcodec": "libwebp", "qscale": 2, "ext": ".webp"},
    "avif": {"desc": "AVIF (smallest)", "vcodec": "libaom-av1", "crf": 30, "ext": ".avif"},
}

# Presets for resize
RESIZE_PRESETS = {
    "480p": {"width": 854, "height": 480, "desc": "480p (SD)"},
    "720p": {"width": 1280, "height": 720, "desc": "720p (HD)"},
    "1080p": {"width": 1920, "height": 1080, "desc": "1080p (Full HD)"},
    "2k": {"width": 2560, "height": 1440, "desc": "2K (QHD)"},
    "4k": {"width": 3840, "height": 2160, "desc": "4K (UHD)"},
    "square-512": {"width": 512, "height": 512, "desc": "512x512 (AI)"},
    "square-768": {"width": 768, "height": 768, "desc": "768x768 (AI)"},
    "square-1024": {"width": 1024, "height": 1024, "desc": "1024x1024 (AI)"},
}


def get_media_info(path: Path) -> dict:
    """Get media file info using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        logger.warning(f"ffprobe failed: {e}")
    return {}


def convert_video(
    input_path: Path,
    preset: str = "mp4-h264",
    output_path: Optional[Path] = None,
    resize: Optional[str] = None,
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
    custom_args: Optional[List[str]] = None,
) -> Tuple[Path, dict]:
    """Convert video with preset."""
    if preset not in VIDEO_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(VIDEO_PRESETS.keys())}")
    
    p = VIDEO_PRESETS[preset]
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_{preset}{p['ext']}"
    
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    
    if trim_start is not None:
        cmd += ["-ss", str(trim_start)]
    
    cmd += ["-i", str(input_path)]
    
    if trim_end is not None:
        cmd += ["-t", str(trim_end - (trim_start or 0))]
    
    # Video codec
    if p.get("vcodec"):
        cmd += ["-c:v", p["vcodec"]]
    if p.get("crf") is not None:
        cmd += ["-crf", str(p["crf"])]
    if p.get("preset"):
        cmd += ["-preset", p["preset"]]
    if p.get("max_size_mb"):
        cmd += ["-fs", f"{int(p['max_size_mb'])}M"]
    
    # Audio codec
    if p.get("acodec"):
        cmd += ["-c:a", p["acodec"]]
    if p.get("audio_bitrate"):
        cmd += ["-b:a", p["audio_bitrate"]]
    
    # Resize
    if resize and resize in RESIZE_PRESETS:
        r = RESIZE_PRESETS[resize]
        cmd += ["-vf", f"scale={r['width']}:{r['height']}:force_original_aspect_ratio=decrease,pad={r['width']}:{r['height']}:(ow-iw)/2:(oh-ih)/2"]
    elif resize:
        # Custom WxH
        if "x" in resize.lower():
            w, h = resize.lower().split("x")
            cmd += ["-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"]
    
    # GIF special handling
    if preset == "gif":
        duration = p.get("max_duration", 15)
        scale = p.get("scale", 480)
        cmd += ["-vf", f"fps=10,scale={scale}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse", "-loop", "0"]
        cmd += ["-t", str(duration)]
    
    if custom_args:
        cmd += custom_args
    
    cmd += [str(output_path)]
    
    logger.info(f"Converting: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e}")
    
    info = get_media_info(output_path)
    return output_path, info


def convert_audio(
    input_path: Path,
    preset: str = "mp3-128",
    output_path: Optional[Path] = None,
    trim_start: Optional[float] = None,
    trim_end: Optional[float] = None,
    custom_args: Optional[List[str]] = None,
) -> Tuple[Path, dict]:
    """Convert audio with preset."""
    if preset not in AUDIO_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(AUDIO_PRESETS.keys())}")
    
    p = AUDIO_PRESETS[preset]
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_{preset}{p['ext']}"
    
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    
    if trim_start is not None:
        cmd += ["-ss", str(trim_start)]
    
    cmd += ["-i", str(input_path)]
    
    if trim_end is not None:
        cmd += ["-t", str(trim_end - (trim_start or 0))]
    
    cmd += ["-c:a", p["acodec"]]
    if p.get("audio_bitrate"):
        cmd += ["-b:a", p["audio_bitrate"]]
    if p.get("ar"):
        cmd += ["-ar", str(p["ar"])]
    if p.get("ac"):
        cmd += ["-ac", str(p["ac"])]
    
    if custom_args:
        cmd += custom_args
    
    cmd += [str(output_path)]
    
    logger.info(f"Converting audio: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio conversion failed: {e}")
    
    info = get_media_info(output_path)
    return output_path, info


def extract_audio(
    input_path: Path,
    output_path: Optional[Path] = None,
    audio_format: str = "wav",
) -> Tuple[Path, dict]:
    """Extract audio from video file."""
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_audio.{audio_format}"

    if audio_format == "wav":
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(output_path)
        ]
    elif audio_format in ("mp3",):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-vn", "-c:a", "libmp3lame", "-b:a", "128k",
            str(output_path)
        ]
    elif audio_format in ("aac", "m4a"):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-vn", "-c:a", "aac", "-b:a", "128k",
            str(output_path)
        ]
    elif audio_format == "opus":
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-vn", "-c:a", "libopus", "-b:a", "96k",
            str(output_path)
        ]
    elif audio_format == "flac":
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-vn", "-c:a", "flac",
            str(output_path)
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path),
            "-vn", "-c:a", "pcm_s16le",
            str(output_path)
        ]

    logger.info(f"Extracting audio: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Audio extraction failed: {e}")

    info = get_media_info(output_path)
    return output_path, info


def trim_media(
    input_path: Path,
    start: float,
    end: float,
    output_path: Optional[Path] = None,
) -> Tuple[Path, dict]:
    """Trim media file to specified time range."""
    if output_path is None:
        ext = input_path.suffix
        output_path = input_path.parent / f"{input_path.stem}_trim_{start:.0f}-{end:.0f}{ext}"
    
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(input_path),
        "-c", "copy",
        str(output_path)
    ]
    
    logger.info(f"Trimming: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Trim failed: {e}")
    
    info = get_media_info(output_path)
    return output_path, info


def compress_video(
    input_path: Path,
    target_size_mb: float = 10,
    output_path: Optional[Path] = None,
) -> Tuple[Path, dict]:
    """Compress video to target size using two-pass encoding."""
    info = get_media_info(input_path)
    duration = float(info.get("format", {}).get("duration", 0))
    if duration <= 0:
        raise ValueError("Could not determine video duration")
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_compressed.mp4"
    
    # Calculate target bitrate
    target_bits = target_size_mb * 1024 * 1024 * 8
    audio_bitrate = 128000  # 128 kbps
    video_bitrate = (target_bits / duration) - audio_bitrate
    video_bitrate = max(video_bitrate, 100000)  # min 100 kbps
    
    # Two-pass encoding; keep the pass log next to the temporary output so it
    # never lands in the server's working directory.
    passlog = str(output_path.with_name(output_path.stem + "_passlog"))
    pass1 = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-c:v", "libx264", "-b:v", str(int(video_bitrate)),
        "-pass", "1", "-passlogfile", passlog, "-an", "-f", "null", os.devnull
    ]
    
    pass2 = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-c:v", "libx264", "-b:v", str(int(video_bitrate)),
        "-c:a", "aac", "-b:a", "128k",
        "-pass", "2", "-passlogfile", passlog, str(output_path)
    ]
    
    logger.info(f"Compressing to ~{target_size_mb}MB: {video_bitrate/1000:.0f}kbps video")
    try:
        subprocess.run(pass1, check=True, timeout=300)
        subprocess.run(pass2, check=True, timeout=300)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compression failed: {e}")
    
    info = get_media_info(output_path)
    return output_path, info


def convert_image(
    input_path: Path,
    preset: str = "jpg-95",
    output_path: Optional[Path] = None,
) -> Tuple[Path, dict]:
    """Convert a single image with an image preset."""
    if preset not in IMAGE_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(IMAGE_PRESETS.keys())}")
    p = IMAGE_PRESETS[preset]
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_{preset}{p['ext']}"
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path),
           "-frames:v", "1", "-c:v", p["vcodec"]]
    if p.get("qscale") is not None:
        cmd += ["-q:v", str(p["qscale"])]
    if p.get("crf") is not None:
        cmd += ["-crf", str(p["crf"])]
    cmd += [str(output_path)]
    logger.info(f"Converting image: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, timeout=120)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg image conversion failed: {e}")
    return output_path, get_media_info(output_path)


def get_presets() -> dict:
    """Return all available presets."""
    return {
        "video": VIDEO_PRESETS,
        "audio": AUDIO_PRESETS,
        "image": IMAGE_PRESETS,
        "resize": RESIZE_PRESETS,
    }