"""Unified WebUI must have 4 tabs: Generate, Edit/Inpaint, Upscale, Transcribe."""
from fastapi.testclient import TestClient
import unified_server as us


def test_webui_has_unified_tabs():
    c = TestClient(us.app)
    r = c.get("/webui")
    assert r.status_code == 200
    html = r.text
    for tab in ("Transcribe", "Upscale", "Inpaint", "Generate", "Caption", "Tools", "Models", "Settings"):
        assert tab in html, f"missing tab {tab}"
