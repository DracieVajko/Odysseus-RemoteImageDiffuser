"""Unified contract tests — the Odysseus contract must hold for the combined server.

Cesty z docs/ODYSSEUS-CONTRACT.md + WhisperApi:
GET /health, GET /v1/models, POST /v1/images/generations|edits,
POST /v1/images/inpaint|harmonize|upscale,
POST /v1/audio/transcriptions|translations,
GET /v1/images/progress/{request_id}, GET /webui
"""
import importlib


def _routes():
    mod = importlib.import_module("unified_server")
    paths = set()
    for r in mod.app.routes:
        p = getattr(r, "path", "")
        methods = ",".join(sorted(getattr(r, "methods", set()) or []))
        if p:
            paths.add(f"{methods} {p}")
    return paths


def _has(method, path):
    return any(p.startswith(method) and path in p for p in _routes())


def test_unified_contract_paths_exist():
    assert _has("GET", "/health"), "missing GET /health"
    assert _has("GET", "/v1/models"), "missing GET /v1/models"
    assert _has("POST", "/v1/images/generations"), "missing generations"
    assert _has("POST", "/v1/images/edits"), "missing edits"
    assert _has("POST", "/v1/images/inpaint"), "missing inpaint"
    assert _has("POST", "/v1/images/harmonize"), "missing harmonize"
    assert _has("POST", "/v1/images/upscale"), "missing upscale (new in V1)"
    assert _has("POST", "/v1/audio/transcriptions"), "missing transcriptions"
    assert _has("POST", "/v1/audio/translations"), "missing translations"
    assert _has("GET", "/v1/images/progress"), "missing progress poll"


def test_upscale_contract_shape():
    import base64
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    import unified_server as us

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    c = TestClient(us.app)
    r = c.post("/v1/images/upscale", json={"image": b64, "scale": 2})
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    assert "image" in body and "elapsed" in body, body.keys()


def test_audio_transcription_contract_shape():
    from fastapi.testclient import TestClient
    import unified_server as us

    c = TestClient(us.app)
    # an empty upload must return 400, not 500 (contract)
    r = c.post("/v1/audio/transcriptions", files={"audio": ("e.wav", b"", "audio/wav")})
    assert r.status_code in (400, 415, 422), r.text[:300]
