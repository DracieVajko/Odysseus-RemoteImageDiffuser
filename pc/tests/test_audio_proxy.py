"""Audio proxy — unified_server has no faster-whisper in its own env,
so real transcription delegates to the running WhisperApi service (:4444).
Without it, returns None and the endpoint takes the stub path (the contract shape holds)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from unified.audio_proxy import transcribe_via_service


class _Stub(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        body = json.dumps({"text": "ahoj svet", "language": "sk"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_proxy_forwards_to_service():
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = transcribe_via_service(
            f"http://127.0.0.1:{srv.server_port}",
            b"RIFF" + b"\x00" * 64, "t.wav", "small", "sk",
        )
        assert out == ("ahoj svet", "sk"), out
    finally:
        srv.shutdown()


def test_proxy_none_on_refused():
    out = transcribe_via_service("http://127.0.0.1:1", b"x", "t.wav", "small", None)
    assert out is None
