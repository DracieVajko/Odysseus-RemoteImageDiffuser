"""FFmpeg utilities for audio processing (splitting, conversion, etc.)."""
import logging
import subprocess
import tempfile
import math
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Whisper context limit (seconds) - we use 25s segments with 5s overlap for safety
WHISPER_CHUNK_SECONDS = 25
WHISPER_OVERLAP_SECONDS = 5

# Max duration before we auto-split (seconds) - Whisper context is 30s
AUTO_SPLIT_THRESHOLD_SECONDS = 30


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"ffprobe failed: {e}")
    return 0.0


def split_audio_with_ffmpeg(
    input_path: Path,
    chunk_seconds: int = WHISPER_CHUNK_SECONDS,
    overlap_seconds: int = WHISPER_OVERLAP_SECONDS,
    output_dir: Optional[Path] = None,
    keep_original: bool = True
) -> List[Tuple[Path, float, float]]:
    """
    Split audio file into overlapping chunks using FFmpeg.
    
    Returns list of (chunk_path, start_time, end_time).
    """
    if output_dir is None:
        output_dir = input_path.parent
    
    duration = get_audio_duration(input_path)
    if duration <= 0:
        logger.warning(f"Could not determine duration of {input_path}")
        return [(input_path, 0.0, 0.0)]
    
    # If short enough, no need to split
    if duration <= AUTO_SPLIT_THRESHOLD_SECONDS:
        return [(input_path, 0.0, duration)]
    
    logger.info(f"Splitting {duration:.1f}s audio into {chunk_seconds}s chunks with {overlap_seconds}s overlap")
    
    chunks = []
    chunk_dir = output_dir / f"{input_path.stem}_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    step = chunk_seconds - overlap_seconds
    num_chunks = math.ceil((duration - overlap_seconds) / step)
    
    for i in range(num_chunks):
        start = i * step
        if start >= duration:
            break
        
        end = min(start + chunk_seconds, duration)
        actual_duration = end - start
        
        chunk_path = chunk_dir / f"chunk_{i:03d}_{start:.1f}-{end:.1f}.wav"
        
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-t", f"{actual_duration:.3f}",
            "-i", str(input_path),
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            str(chunk_path)
        ]
        
        try:
            subprocess.run(cmd, check=True, timeout=60)
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                chunks.append((chunk_path, start, end))
                logger.debug(f"Created chunk {i}: {start:.1f}-{end:.1f}s")
            else:
                logger.warning(f"Chunk {i} is empty")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed for chunk {i}: {e}")
            raise RuntimeError(f"Failed to create chunk {i}: {e}")
    
    logger.info(f"Created {len(chunks)} chunks")
    return chunks


def transcribe_chunks(
    chunks: List[Tuple[Path, float, float]],
    transcribe_fn,
    language: Optional[str] = None
) -> Tuple[str, str]:
    """
    Transcribe multiple chunks and combine results with timestamps.
    
    Returns (combined_text, language).
    """
    if len(chunks) == 1:
        text, lang = transcribe_fn(chunks[0][0], language)
        return text, lang
    
    all_text = []
    detected_lang = None
    
    for i, (chunk_path, start, end) in enumerate(chunks):
        try:
            text, lang = transcribe_fn(chunk_path, language)
            if lang and not detected_lang:
                detected_lang = lang
            
            # Add timestamp prefix
            timestamp = format_timestamp(start)
            if text.strip():
                all_text.append(f"[{timestamp}] {text.strip()}")
        except Exception as e:
            logger.error(f"Failed to transcribe chunk {i}: {e}")
            all_text.append(f"[{format_timestamp(start)}] [ERROR: {e}]")
    
    return "\n".join(all_text), detected_lang


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def cleanup_chunks(chunks: List[Tuple[Path, float, float]], keep_original: bool = True):
    """Clean up temporary chunk files."""
    for chunk_path, _, _ in chunks:
        if chunk_path.name.startswith("chunk_"):
            try:
                chunk_path.unlink(missing_ok=True)
            except Exception:
                pass
        # Also try to remove parent dir if empty
        try:
            parent = chunk_path.parent
            if parent.name.endswith("_chunks") and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass


# Integration helper for unified_server
async def transcribe_with_splitting(
    audio_path: Path,
    transcribe_fn,
    language: Optional[str] = None,
    max_duration_without_split: float = AUTO_SPLIT_THRESHOLD_SECONDS
) -> Tuple[str, str]:
    """
    High-level function: transcribe audio, auto-splitting if needed.
    
    Args:
        audio_path: Path to audio file
        transcribe_fn: Function taking (path, language) -> (text, lang)
        language: Optional language code
        max_duration_without_split: Max seconds before auto-splitting
    
    Returns: (combined_text, language)
    """
    duration = get_audio_duration(audio_path)
    if duration <= max_duration_without_split:
        return transcribe_fn(audio_path, language)
    
    logger.info(f"Audio duration {duration:.1f}s exceeds threshold, splitting...")
    chunks = split_audio_with_ffmpeg(audio_path)
    try:
        return transcribe_chunks(chunks, transcribe_fn, language)
    finally:
        cleanup_chunks(chunks)