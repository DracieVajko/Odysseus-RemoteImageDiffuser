"""ffmpeg_pre gate — validácia skôr ako drahá inferencia."""
import pytest
from pathlib import Path
from unified.ffmpeg_pre import (
    normalize_audio_to_wav16k, UnsupportedFormat, EmptyFile, TooLarge,
)


def test_rejects_unsupported_ext(tmp_path):
    src = tmp_path / "x.xyz"
    src.write_bytes(b"abc")
    with pytest.raises(UnsupportedFormat):
        normalize_audio_to_wav16k(src, tmp_path / "o.wav")


def test_rejects_empty_file(tmp_path):
    src = tmp_path / "e.wav"
    src.write_bytes(b"")
    with pytest.raises(EmptyFile):
        normalize_audio_to_wav16k(src, tmp_path / "o.wav")


def test_passthrough_wav(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(b"RIFF" + b"\x00" * 100)
    out = normalize_audio_to_wav16k(src, tmp_path / "o.wav")
    assert Path(out["path"]).exists()
