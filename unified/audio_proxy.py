"""Forward transcription to a running WhisperApi service (own faster-whisper env)."""
import io
import json
import logging
import urllib.request

logger = logging.getLogger("unified.audio_proxy")


def transcribe_via_service(base_url, content: bytes, filename: str,
                           model: str, language, timeout=600):
    """POST multipart na {base}/v1/audio/transcriptions. (text, lang)|None."""
    bound = "----unifiedaudio"
    body = io.BytesIO()
    body.write(f'--{bound}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode())
    if language:
        body.write(f'--{bound}\r\nContent-Disposition: form-data; name="language"\r\n\r\n{language}\r\n'.encode())
    body.write(f'--{bound}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
               f'Content-Type: application/octet-stream\r\n\r\n'.encode())
    body.write(content + b"\r\n")
    body.write(f"--{bound}--\r\n".encode())
    try:
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/audio/transcriptions",
            data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={bound}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return data.get("text", ""), data.get("language", language)
    except Exception as e:
        logger.warning(f"audio proxy {base_url} failed: {str(e)[:140]}")
        return None
