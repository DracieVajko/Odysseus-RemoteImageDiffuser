"""Model install/uninstall manager for PC (Windows/Mac/Linux).

Shares the catalog with download_models.py (TASKS + UPSCALE_TASKS + VLM_TASKS).
Install = background download (thread), uninstall = safe deletion of the model
directory. Progress is an in-memory dict queryable over HTTP.

Same behavior as the Android ModelDownloader: without an installed model,
inference returns 503 will-not-load; uninstall never deletes anything outside models/.

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

# Install state: id -> {"state": idle|downloading|done|error, "progress": 0-100, ...}
_install_state: dict = {}
_state_lock = threading.Lock()

KIND_LABELS = {
    "image": "Image generation",
    "audio": "Transcription (Whisper)",
    "upscale": "Upscale",
    "vlm": "Vision-language (VLM)",
    "custom": "Custom (own URL)",
}


def _catalog_tasks():
    """Import the catalog from download_models.py (single source of truth)."""
    import download_models as dm
    out = []
    out += [dict(t, group="image") for t in dm.TASKS]
    out += [dict(t, group="upscale") for t in dm.UPSCALE_TASKS]
    out += [dict(t, group="vlm") for t in dm.VLM_TASKS]
    return out


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
    """A model is installed if it has enough data (min_gb) or is a
    single-file model with an existing file."""
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
        group = t.get("group", "image")
        out.append({
            "id": t["id"],
            "kind": group,
            "kind_label": KIND_LABELS.get(group, group),
            "engine": "url" if t.get("kind") == "url" else "torch",
            "label": t.get("note") or t.get("repo") or t.get("url") or t["id"],
            "license": t.get("license", ""),
            "size_gb": t.get("min_gb", 0.0),
            "size_mb": int(round(t.get("min_gb", 0.0) * 1024)),
            "installed": is_installed(t),
            "path": str(target),
            "installed_gb": round(_dir_gb(target), 2),
            "install_state": st.get("state", "idle"),
            "install_progress": st.get("progress", 0),
            "install_error": st.get("error", ""),
            "has_source": True,
            "downloadable": True,
            "runs_on_device": True,
            "custom": False,
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
        try:
            import download_models as _dm
            if isinstance(e, _dm.CancelledError):
                _set_state(task["id"], state="cancelled", progress=0, error="")
                _dm._CANCEL.discard(task["id"])
                logger.info("install %s cancelled", task["id"])
                return
        except Exception:
            pass
        logger.warning("install %s failed: %s", task["id"], e)
        _set_state(task["id"], state="error", progress=0, error=str(e)[:300])


def start_install(model_id: str) -> dict:
    task = _task(model_id)
    if task is None:
        raise KeyError(f"unknown model id: {model_id}")
    st = _install_state.get(model_id, {})
    if st.get("state") == "downloading":
        return {"id": model_id, "state": "downloading", "already_running": True}
    try:
        import download_models as _dm
        _dm._CANCEL.discard(model_id)
    except Exception:
        pass
    threading.Thread(target=_run_install, args=(task,), daemon=True).start()
    return {"id": model_id, "state": "downloading"}


def install_blocking(model_id: str) -> dict:
    """CLI: download a model synchronously and return the result."""
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


def progress(model_id: str) -> dict:
    """Install state in the same shape as Android /v1/models/progress/{id}."""
    task = _task(model_id)
    if task is None:
        raise KeyError(f"unknown model id: {model_id}")
    st = _install_state.get(model_id, {})
    return {
        "id": model_id,
        "state": st.get("state", "done" if is_installed(task) else "idle"),
        "percent": st.get("progress", 100 if is_installed(task) else 0),
        "bytes_done": 0,
        "bytes_total": int(round(task.get("min_gb", 0.0) * 1024 * 1024 * 1024)),
        "file_index": 0,
        "file_count": 1,
        "error": st.get("error", ""),
    }


def status(model_id: str) -> dict:
    task = _task(model_id)
    if task is None:
        raise KeyError(f"unknown model id: {model_id}")
    target = _target_path(task)
    group = task.get("group", "image")
    return {
        "id": model_id,
        "kind": group,
        "kind_label": KIND_LABELS.get(group, group),
        "engine": "url" if task.get("kind") == "url" else "torch",
        "label": task.get("note") or task.get("repo") or task.get("url") or model_id,
        "license": task.get("license", ""),
        "path": str(target),
        "installed": is_installed(task),
        "installed_gb": round(_dir_gb(target), 2),
        "state": _install_state.get(model_id, {}).get("state", "idle"),
        "error": _install_state.get(model_id, {}).get("error", ""),
    }


def verify(model_id: str) -> dict:
    """PC snapshots have no pinned SHA-256; verification = completeness by min_gb."""
    task = _task(model_id)
    if task is None:
        raise KeyError(f"unknown model id: {model_id}")
    ok = is_installed(task)
    return {
        "id": model_id,
        "verified": ok,
        "method": "size>=min_gb (torch snapshot, no pinned sha256)",
    }


def cancel(model_id: str) -> dict:
    """Request cancellation. URL downloads stop immediately (chunk loop);
    HF snapshots stop between files."""
    if _task(model_id) is None:
        raise KeyError(f"unknown model id: {model_id}")
    running = _install_state.get(model_id, {}).get("state") == "downloading"
    try:
        import download_models as _dm
        _dm._CANCEL.add(model_id)
    except Exception:
        pass
    return {
        "id": model_id,
        "cancelled": True,
        "running": running,
        "reason": "stop requested; URL downloads stop immediately, HF snapshots between files",
    }
