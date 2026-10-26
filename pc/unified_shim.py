#!/usr/bin/env python3
"""Legacy :4444 shim — forwards everything to unified :8101.

Stdlib only (works on system Python without dependencies):
  python unified_shim.py --target http://127.0.0.1:8101 --port 4444
For Odysseus clients that hardcode :4444 from the WhisperApi era.
"""
import argparse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def run_shim(target: str):
    base = target.rstrip("/")

    class ShimHandler(BaseHTTPRequestHandler):
        def _forward(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else None
            headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
            url = base + self.path
            try:
                req = urllib.request.Request(url, data=body, headers=headers,
                                             method=self.command)
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = r.read()
                    self.send_response(r.status)
                    self.send_header("Content-Type",
                                     r.headers.get("Content-Type", "application/json"))
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            except Exception as e:
                msg = f'{{"error": "shim upstream unreachable: {str(e)[:160]}"}}'.encode()
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

        do_GET = _forward
        do_POST = _forward
        do_PUT = _forward
        do_DELETE = _forward

        def log_message(self, *a):
            pass

    return ShimHandler


def main():
    ap = argparse.ArgumentParser(description="Unified legacy shim :4444 -> :8101")
    ap.add_argument("--target", default="http://127.0.0.1:8101")
    ap.add_argument("--port", type=int, default=4444)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), run_shim(args.target))
    print(f"shim {args.host}:{args.port} -> {args.target}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
