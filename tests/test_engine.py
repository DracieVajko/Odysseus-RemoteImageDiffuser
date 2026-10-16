"""Engine init + real inpaint delegation (lazy, no weights at import)."""
from unified.engine import init_diffusion_engine, try_real_inpaint


def test_init_is_safe_without_torch():
    # V sys pythone bez torch nesmie spadnúť, vráti False (engine nedostupný)
    ok = init_diffusion_engine()
    assert ok is False


def test_real_inpaint_none_without_engine():
    assert try_real_inpaint("e30=", "e30=", "x", "DreamShaper_8") is None


def test_real_inpaint_rejects_garbage():
    import pytest
    with pytest.raises(ValueError):
        try_real_inpaint("!!!not-b64!!!", "!!!not-b64!!!", "x", "DreamShaper_8",
                         force_engine=True)
