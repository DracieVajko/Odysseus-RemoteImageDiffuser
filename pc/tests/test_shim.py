"""Legacy :4444 shim — forwards everything to unified :8101 (stdlib, no dependencies)."""
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from unified_shim import run_shim


class _Upstream(BaseHTTPRequestHandler):
    def _ok(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._ok({"status": "ok", "via": "upstream"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n)
        self._ok({"text": "proxied", "language": "sk"})

    def log_message(self, *a):
        pass


def test_shim_forwards_get_and_post():
    up = HTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    shim = HTTPServer(("127.0.0.1", 0), run_shim(f"http://127.0.0.1:{up.server_port}"))
    threading.Thread(target=shim.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{shim.server_port}"
        h = json.loads(urllib.request.urlopen(base + "/v1/health", timeout=10).read())
        assert h["via"] == "upstream", h
        req = urllib.request.Request(base + "/v1/audio/transcriptions", data=b"x", method="POST")
        t = json.loads(urllib.request.urlopen(req, timeout=10).read())
        assert t["text"] == "proxied", t
    finally:
        shim.shutdown()
        up.shutdown()
