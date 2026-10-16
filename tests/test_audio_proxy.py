"""Audio proxy — unified_server nemá faster-whisper vo vlastnom env,
tak reálny prepis deleguje na bežiacu WhisperApi službu (:4444).
Bez nej vráti None a endpoint ide stub cestou (kontrakt drží tvar)."""
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
