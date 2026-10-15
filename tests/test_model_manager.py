"""ModelManager — vždy max 1 ťažký model rezidentný."""
import pytest
from unified.model_manager import ModelManager, ModelLoadingError


def test_only_one_resident():
    mm = ModelManager()
    mm.ensure("image", "DreamShaper_8")
    assert mm.status()["loaded_kind"] == "image"
    mm.ensure("audio", "small")
    st = mm.status()
    assert st["loaded_kind"] == "audio"
    assert st["loaded_id"] == "small"


def test_same_model_no_reload():
    mm = ModelManager()
    mm.ensure("image", "DreamShaper_8")
    n1 = mm.load_count
    mm.ensure("image", "DreamShaper_8")
    assert mm.load_count == n1


def test_loading_503_shape():
    mm = ModelManager()
    mm.set_loading("image", "JuggernautXL_v9", True)
    with pytest.raises(ModelLoadingError) as e:
        mm.ensure("image", "DreamShaper_8")
    assert e.value.retry_after == 30
    assert "loading" in str(e.value.detail).lower()


def test_unload_frees():
    mm = ModelManager()
    mm.ensure("image", "DreamShaper_8")
    assert mm.unload("test") is True
    assert mm.status()["loaded_kind"] is None
    assert mm.unload("test") is False
