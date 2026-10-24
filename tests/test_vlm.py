"""VLM caption — bad input 400, missing model 503, not 500."""
import pytest
from unified.vlm import caption_image_b64, find_vlm_dir, EngineMissing
import base64
import io
from PIL import Image


def _red_dot_b64(size=32):
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (255, 0, 0)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_find_vlm_dir_none_when_absent(tmp_path):
    assert find_vlm_dir([tmp_path]) is None


def test_caption_rejects_garbage():
    with pytest.raises(ValueError):
        caption_image_b64("!!!not-b64!!!")


def test_caption_missing_model_raises(tmp_path):
    with pytest.raises(EngineMissing):
        caption_image_b64(_red_dot_b64(), search_dirs=[tmp_path])


def test_caption_contract_no_model():
    # Without weights the endpoint reports 503 will-not-load (verified via TestClient).
    import os
    os.environ["UNIFIED_NO_ENGINE"] = "1"
    from fastapi.testclient import TestClient
    import unified_server as us
    c = TestClient(us.app)
    r = c.post("/v1/images/caption", json={"image": _red_dot_b64()})
    assert r.status_code == 503, r.text[:200]
    assert r.json()["error"] == "will not load"


def test_chat_completions_no_image_400():
    from fastapi.testclient import TestClient
    import unified_server as us
    c = TestClient(us.app)
    r = c.post("/v1/chat/completions", json={
        "model": "SmolVLM_500M",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert r.status_code == 400, r.text[:200]


def test_chat_completions_vision_no_model_503():
    import os
    os.environ["UNIFIED_NO_ENGINE"] = "1"
    from fastapi.testclient import TestClient
    import unified_server as us
    c = TestClient(us.app)
    img = _red_dot_b64()
    r = c.post("/v1/chat/completions", json={
        "model": "SmolVLM_500M",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
        ]}],
    })
    assert r.status_code == 503, r.text[:200]
    body = r.json()
    assert body["error"] == "will not load"
