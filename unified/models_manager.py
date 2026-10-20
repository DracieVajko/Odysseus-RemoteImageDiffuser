"""Model install/uninstall manager pre PC (Windows/Mac/Linux).

Zdieľa katalóg s download_models.py (TASKS + UPSCALE_TASKS + VLM_TASKS).
Install = stiahnutie na pozadí (thread), uninstall = bezpečné zmazanie
adresára modelu. Progress je in-memory slovník dotazovateľný cez HTTP.

Rovnaké správanie ako Android ModelDownloader: bez nainštalovaného modelu
vracia inference 503 will-not-load; uninstall nikdy nezmaže nič mimo
models/ (path guard).
"""
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("unified.models")

HERE = Path(__file__).resolve().parent.parent
MODELS_DIR = HERE / "models"

# Stav inštalácií: id -> {"state": idle|downloading|done|error, "progress": 0-100, ...}
_install_state: dict = {}
_state_lock = threading.Lock()


def _catalog_tasks():
    """Import katalógu z download_models.py (single source of truth)."""
    import download_models as dm
    return list(dm.TASKS) + list(dm.UPSCALE_TASKS) + list(dm.VLM_TASKS)


def _task(id: str):
    for t in _catalog_tasks():
        if t["id"] == id:
            return t
    return None


def _target_path(task: dict) -> Path:
    return MODELS_DIR / task["target"]


def _dir_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def is_installed(task: dict) -> bool:
    """Model je nainštalovaný, ak má dostatok dát (min_gb) alebo je to
    jednosúborový model s existujúcim súborom."""
    target = _target_path(task)
    if task.get("kind") == "file":
        f = target / task["file"]
        return f.is_file() and f.stat().st_size / 1e9 >= task.get("min_gb", 0.0)
    if task.get("kind") == "url":
        f = target / task["file"]
        return f.is_file() and f.stat().st_size / 1e9 >= task.get("min_gb", 0.0)
    return _dir_gb(target) >= task.get("min_gb", 0.0)


def catalog() -> list:
    out = []
    for t in _catalog_tasks():
        target = _target_path(t)
        st = _install_state.get(t["id"], {})
        out.append({
            "id": t["id"],
            "kind": t.get("kind", "snapshot"),
            "label": t.get("note") or t.get("repo") or t.get("url") or t["id"],
            "license": t.get("license", ""),
            "size_gb": t.get("min_gb", 0.0),
            "installed": is_installed(t),
            "path": str(target),
            "installed_gb": round(_dir_gb(target), 2),
            "install_state": st.get("state", "idle"),
            "install_progress": st.get("progress", 0),
            "install_error": st.get("error", ""),
            "has_source": True,
        })
    return out


def _set_state(model_id: str, **kw):
    with _state_lock:
        cur = _install_state.setdefault(model_id, {"state": "idle", "progress": 0, "error": ""})
        cur.update(kw)


def _run_install(task: dict):
    import download_models as dm
    try:
        _set_state(task["id"], state="downloading", progress=1, error="")
        if task.get("kind") == "url":
            dm.download_url(task)
        else:
            dm.download_hf(task)
        ok = is_installed(task)
        _set_state(task["id"], state="done" if ok else "error",
                   progress=100, error="" if ok else "download incomplete")
        logger.info("install %s -> %s", task["id"], "done" if ok else "error")
    except Exception as e:
        logger.warning("install %s failed: %s", task["id"], e)
        _set_state(task["id"], state="error", progress=0, error=str(e)[:300])


def start_install(model_id: str) -> dict:
    task = _task(model_id)
    if task is None:
        raise KeyError(f"unknown model id: {model_id}")
    st = _install_state.get(model_id, {})
    if st.get("state") == "downloading":
        return {"id": model_id, "state": "downloading", "already_running": True}
    threading.Thread(target=_run_install, args=(task,), daemon=True).start()
    return {"id": model_id, "state": "downloading"}


def install_blocking(model_id: str) -> dict:
    """CLI: stiahne model synchrónne a vráti výsledok."""
    task = _task(model_id)
    if task is None:
        raise KeyError(f"unknown model id: {model_id}")
    _run_install(task)
    st = _install_state.get(model_id, {})
    return {"id": model_id, "state": st.get("state", "idle"),
            "installed": is_installed(task), "error": st.get("error", "")}


def uninstall(model_id: str) -> dict:
    task = _task(model_id)
    if task is None:
        raise KeyError(f"unknown model id: {model_id}")
    target = _target_path(task).resolve()
    guard = MODELS_DIR.resolve()
    if guard not in target.parents and target != guard:
        raise ValueError(f"refusing to delete outside models/: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.is_file():
        target.unlink()
    # model-specific extra: single file inside target dir
    if task.get("kind") == "file":
        f = _target_path(task) / task["file"]
        if f.is_file():
            f.unlink()
    _install_state.pop(model_id, None)
    logger.info("uninstalled %s (%s)", model_id, target)
    return {"id": model_id, "removed": not target.exists()}
