"""Engine init + real inpaint delegation (lazy, no weights at import).

The engine is optional: without torch/diffusers init must fail cleanly, with
them installed it must initialise the registry. The tests below therefore key
off what is actually installed instead of assuming one environment.
"""
import pytest

import unified.engine as engine
import diffusion_server as ds


def _stack_available() -> bool:
    return engine._have_torch()


def test_init_matches_installed_stack():
    ok = engine.init_diffusion_engine()
    if _stack_available():
        assert ok is True
        # init must have loaded models.json into the registry (no weights)
        assert ds._REGISTRY, "registry empty after init_diffusion_engine"
    else:
        assert ok is False


def test_real_inpaint_none_without_engine(monkeypatch):
    # Force the "no engine" state so the fallback contract is deterministic,
    # independent of whether torch/diffusers happen to be installed.
    monkeypatch.setattr(engine, "_is_engine_ready", lambda: False)
    assert engine.try_real_inpaint("e30=", "e30=", "x", "DreamShaper_8") is None


def test_real_inpaint_rejects_garbage():
    with pytest.raises(ValueError):
        engine.try_real_inpaint("!!!not-b64!!!", "!!!not-b64!!!", "x",
                                "DreamShaper_8", force_engine=True)
