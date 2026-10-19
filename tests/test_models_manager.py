"""Model manager (PC): katalóg, installed detekcia, uninstall guard."""
import pytest
from pathlib import Path

import unified.models_manager as mm


def test_catalog_has_known_models():
    rows = mm.catalog()
    ids = {r["id"] for r in rows}
    assert "DreamShaper_8" in ids
    assert "RealESRGAN_x4v3" in ids
    assert "SmolVLM_500M" in ids
    for r in rows:
        assert "installed" in r and "install_state" in r and "size_gb" in r


def test_is_installed_for_url_kind(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
    task = {"id": "t", "kind": "url", "target": "T", "file": "w.bin", "min_gb": 0.001}
    assert mm.is_installed(task) is False
    d = tmp_path / "T"
    d.mkdir()
    (d / "w.bin").write_bytes(b"\0" * 2000)  # 2 KB >= 0.001 GB? no
    assert mm.is_installed(task) is False
    (d / "w.bin").write_bytes(b"\0" * 2_000_000)  # 2 MB > 1 MB
    task["min_gb"] = 0.001
    assert mm.is_installed(task) is True


def test_uninstall_refuses_unknown_id():
    with pytest.raises(KeyError):
        mm.uninstall("definitely-not-a-model")


def test_uninstall_refuses_outside_models_dir(tmp_path, monkeypatch):
    # task s target obsahujúcim .. nesmie zmazať nič mimo MODELS_DIR
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir()
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep")
    monkeypatch.setattr(mm, "_task", lambda _id: {"id": "x", "kind": "snapshot",
                                                  "target": "../victim", "min_gb": 0})
    with pytest.raises(ValueError):
        mm.uninstall("x")
    assert (victim / "keep.txt").exists()


def test_uninstall_removes_model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(mm, "MODELS_DIR", tmp_path)
    d = tmp_path / "Demo"
    d.mkdir()
    (d / "a.bin").write_bytes(b"x")
    monkeypatch.setattr(mm, "_task", lambda _id: {"id": "demo", "kind": "snapshot",
                                                  "target": "Demo", "min_gb": 0})
    res = mm.uninstall("demo")
    assert res["removed"] is True
    assert not d.exists()
