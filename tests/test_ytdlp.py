"""yt-dlp wrapper — 400 pri zlom URL, 503 bez yt-dlp, 200 pri úspechu."""
import pytest
from unified.ytdlp import download_audio, YtDlpMissing, DownloadFailed


def test_missing_ytdlp(monkeypatch):
    monkeypatch.setattr("unified.ytdlp._have_ytdlp", lambda: False)
    with pytest.raises(YtDlpMissing):
        download_audio("https://example.com/x", tmpdir="/tmp")


def test_invalid_url():
    with pytest.raises(DownloadFailed):
        download_audio("not-a-url", tmpdir="/tmp")


def test_download_audio_success(monkeypatch, tmp_path):
    def fake(url, tmpdir):
        out = tmp_path / "out.wav"
        out.write_bytes(b"RIFF" + b"\x00" * 64)
        return str(out)
    monkeypatch.setattr("unified.ytdlp._have_ytdlp", lambda: True)
    monkeypatch.setattr("unified.ytdlp._run", fake)
    p = download_audio("https://example.com/x.wav", tmpdir=str(tmp_path))
    assert p.endswith(".wav") and (tmp_path / "out.wav").exists()