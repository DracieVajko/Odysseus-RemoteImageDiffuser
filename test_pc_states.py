"""PC /health state test: busy, queue and in_flight must follow real requests.

Requires a running server:
    python -m uvicorn unified_server:app --host 127.0.0.1 --port 8200
"""
import os
import threading
import time

import httpx

BASE = os.getenv("PRISM_BASE", "http://127.0.0.1:8200")
c = httpx.Client(base_url=BASE, timeout=900.0)
failures = []


def health():
    return c.get("/health").json()


def expect(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        failures.append(name)


def wait_for(predicate, timeout=60.0, interval=0.5):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        last = health()
        if predicate(last):
            return last
        time.sleep(interval)
    return last


def gen(tag, sizes, out):
    try:
        r = c.post("/v1/images/generations", json={
            "prompt": "a red cube on a wooden table",
            "size": sizes, "quality": "high", "n": 1,
        })
        out[tag] = r.status_code
    except Exception as e:  # pragma: no cover
        out[tag] = f"EXC {e}"


print("=== BASELINE ===")
h = health()
expect("idle at start", h["busy"] is False and h["queue"] == 0 and h["in_flight"] == 0,
       f"busy={h['busy']} queue={h['queue']} in_flight={h['in_flight']}")

print("\n=== SINGLE REQUEST ===")
out = {}
t1 = threading.Thread(target=gen, args=("a", "512x512", out))
t1.start()
h = wait_for(lambda s: s["busy"] is True and s["in_flight"] >= 1, timeout=20)
expect("busy during request", h["busy"] is True, f"in_flight={h['in_flight']} queue={h['queue']}")
expect("queue empty with one request", h["queue"] == 0, f"queue={h['queue']}")

print("\n=== TWO CONCURRENT REQUESTS ===")
t2 = threading.Thread(target=gen, args=("b", "512x512", out))
t2.start()
h = wait_for(lambda s: s["queue"] >= 1, timeout=30)
expect("queue grows with a second request", h["queue"] >= 1,
       f"busy={h['busy']} queue={h['queue']} in_flight={h['in_flight']}")
t1.join(600)
t2.join(600)
expect("both requests completed", out.get("a") == 200 and out.get("b") == 200, f"statuses={out}")

h = wait_for(lambda s: s["in_flight"] == 0, timeout=20)
expect("idle after completion",
       h["busy"] is False and h["queue"] == 0 and h["in_flight"] == 0,
       f"busy={h['busy']} queue={h['queue']} in_flight={h['in_flight']}")

print("\n=== FAILED REQUEST ===")
r = c.post("/v1/images/inpaint", json={
    "image": "!!!not-base64!!!", "mask": "!!!not-base64!!!",
    "prompt": "x", "width": 256, "height": 256, "mode": "add",
})
expect("invalid request rejected", r.status_code >= 400, f"status={r.status_code}")
h = wait_for(lambda s: s["in_flight"] == 0, timeout=20)
expect("state reset after failure",
       h["busy"] is False and h["queue"] == 0 and h["in_flight"] == 0,
       f"busy={h['busy']} queue={h['queue']} in_flight={h['in_flight']}")

print("\n=== SUMMARY ===")
if failures:
    print(f"FAIL ({len(failures)}): {failures}")
    raise SystemExit(1)
print("PASS: PC health states match real request flow")
