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
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger("unified.models")

HERE = Path(__file__).resolve().parent.parent
MODELS_DIR = HERE / "models"

# Install state: id -> {"state": idle|downloading|done|error, "progress": 0-100, ...}
_install_state: dict = {}
_state_lock = threading.Lock()

# faster-whisper (CTranslate2) models. Only sources that answered HTTP 200 with
# a known content-length are listed, and every file carries its exact size so a
# truncated download is detected instead of being treated as installed.
# Sizes verified with HEAD requests on 2026-10-26.
WHISPER_SOURCES = {
    "tiny": {
        "repo": "Systran/faster-whisper-tiny",
        "files": {
            "model.bin": 75538270,
            "config.json": 2249,
            "tokenizer.json": 2203239,
            "vocabulary.txt": 459861,
        },
    },
    "base": {
        "repo": "Systran/faster-whisper-base",
        "files": {
            "model.bin": 145217532,
            "config.json": 2249,
            "tokenizer.json": 2203239,
            "vocabulary.txt": 459861,
        },
    },
    "small": {
        "repo": "Systran/faster-whisper-small",
        "files": {
            "model.bin": 483546902,
            "config.json": 2370,
            "tokenizer.json": 2203239,
            "vocabulary.txt": 459861,
        },
    },
    "medium": {
        "repo": "Systran/faster-whisper-medium",
        "files": {
            "model.bin": 1527906378,
            "config.json": 2257,
            "tokenizer.json": 2203239,
            "vocabulary.txt": 459861,
        },
    },
    "large": {
        "repo": "Systran/faster-whisper-large-v3",
        "files": {
            "model.bin": 3087284237,
            "config.json": 2394,
            "tokenizer.json": 2480617,
            "vocabulary.json": 1068114,
            "preprocessor_config.json": 340,
        },
    },
    "turbo": {
        # Systran/faster-whisper-large-v3-turbo is not public (HTTP 401);
        # this repo holds the same CTranslate2 weights (sha256 checked).
        "repo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "files": {
            "model.bin": 1617884929,
            "config.json": 2263,
            "tokenizer.json": 2710337,
            "vocabulary.json": 1068114,
            "preprocessor_config.json": 340,
        },
        "sha256": "e76620f83d5f5b69efd3d87e3dc180c1bd21df9fbebacfd4335e5e1efcc018da",
    },
}
for _src in WHISPER_SOURCES.values():
    _src["model_bin"] = _src["files"]["model.bin"]

_WHISPER_HOSTS = ("https://huggingface.co", "https://hf-mirror.com")


def whisper_dir(name: str) -> Path:
    return MODELS_DIR / f"faster-whisper-{name}"


def whisper_installed(name: str) -> bool:
    src = WHISPER_SOURCES.get(name)
    if not src:
        return False
    target = whisper_dir(name)
    bin_path = target / "model.bin"
    if not bin_path.is_file() or bin_path.stat().st_size != src["model_bin"]:
        return False
    # config.json must parse and tokenizer.json must exist; a 15-byte
    # "Entry not found" body must never count as a model file.
    cfg = target / "config.json"
    if not cfg.is_file():
        return False
    try:
        json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return False
    tok = target / "tokenizer.json"
    return tok.is_file() and tok.stat().st_size > 1024


def whisper_task(name: str) -> Optional[dict]:
    src = WHISPER_SOURCES.get(name)
    if not src:
        return None
    return {
        "id": name,
        "kind": "whisper",
        "group": "audio",
        "target": f"faster-whisper-{name}",
        "repo": src["repo"],
        "min_gb": src["model_bin"] / 1e9,
        "note": f"Whisper {name}",
        "license": "MIT",
    }


def whisper_entry(name: str) -> dict:
    task = whisper_task(name)
    st = _install_state.get(name, {})
    installed = whisper_installed(name)
    return {
        "id": name,
        "kind": "audio",
        "kind_label": KIND_LABELS["audio"],
        "engine": "faster-whisper",
        "label": task["note"],
        "license": task["license"],
        "size_gb": task["min_gb"],
        "size_mb": int(round(task["min_gb"] * 1024)),
        "installed": installed,
        "path": str(whisper_dir(name)),
        "installed_gb": round(_dir_gb(whisper_dir(name)), 2),
        "install_state": st.get("state", "idle"),
        "install_progress": st.get("progress", 0),
        "install_error": st.get("error", ""),
        "has_source": True,
        "downloadable": True,
        "runs_on_device": True,
        "custom": False,
        "repo": task["repo"],
    }


class _Cancelled(Exception):
    pass


def _whisper_cancelled(model_id: str) -> bool:
    return bool(_install_state.get(model_id, {}).get("cancel"))


def _http_open(url: str, start: int = 0, timeout: int = 60):
    req = urllib.request.Request(url, headers={
        "User-Agent": "prism/1.0",
        **({"Range": f"bytes={start}-"} if start else {}),
    })
    return urllib.request.urlopen(req, timeout=timeout)


def _download_whisper_file(model_id: str, repo: str, rel: str, dest: Path,
                           expected: Optional[int], progress_span: tuple,
                           verify_json: bool = False) -> None:
    """Resume a single file across mirrors, verifying the exact size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    lo, hi = progress_span
    have = dest.stat().st_size if dest.is_file() else 0
    if expected is not None and have == expected:
        _set_state(model_id, progress=hi)
        return
    if expected is not None and have > expected:
        dest.unlink()
        have = 0

    last_err = ""
    for attempt in range(4):
        host = _WHISPER_HOSTS[attempt % len(_WHISPER_HOSTS)]
        url = f"{host}/{repo}/resolve/main/{rel}"
        try:
            resp = _http_open(url, start=have)
            mode = "ab" if have else "wb"
            written = have
            with resp, open(dest, mode) as fh:
                while True:
                    if _whisper_cancelled(model_id):
                        raise _Cancelled()
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    if expected:
                        pct = lo + (hi - lo) * min(written / expected, 1.0)
                        _set_state(model_id, progress=int(pct))
            size = dest.stat().st_size
            if expected is not None and size != expected:
                last_err = f"{rel}: got {size} bytes, expected {expected}"
                logger.warning("whisper %s %s", model_id, last_err)
                if size > expected or attempt == 3:
                    dest.unlink(missing_ok=True)
                    have = 0
                else:
                    have = size
                time.sleep(2)
                continue
            if verify_json:
                json.loads(dest.read_text(encoding="utf-8"))
            _set_state(model_id, progress=hi)
            return
        except _Cancelled:
            raise
        except Exception as e:
            last_err = f"{rel}: {str(e)[:150]}"
            logger.warning("whisper %s download attempt %s failed: %s",
                           model_id, attempt + 1, last_err)
            time.sleep(2)
            if dest.is_file():
                have = dest.stat().st_size
                if expected is not None and have > expected:
                    dest.unlink(missing_ok=True)
                    have = 0
    raise RuntimeError(last_err or "download failed")


def _run_whisper_install(name: str) -> None:
    src = WHISPER_SOURCES[name]
    target = whisper_dir(name)
    files = src["files"]
    total = sum(files.values())
    try:
        _set_state(name, state="downloading", progress=1, error="", cancel=False)
        done_before = 0
        for rel, size in files.items():
            # Progress span proportional to the real file size, so a 1.6 GB
            # model.bin does not saturate the bar at 92 % for the whole run.
            lo = 1 + 97 * done_before / total
            hi = 1 + 97 * (done_before + size) / total
            _download_whisper_file(name, src["repo"], rel, target / rel, size,
                                   (lo, hi),
                                   verify_json=rel in ("config.json", "preprocessor_config.json"))
            done_before += size
        digest = src.get("sha256")
        if digest:
            import hashlib
            h = hashlib.sha256()
            with open(target / "model.bin", "rb") as fh:
                for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != digest:
                raise RuntimeError(
                    "model.bin sha256 mismatch (expected %s, got %s)"
                    % (digest[:12], h.hexdigest()[:12]))
        ok = whisper_installed(name)
        _set_state(name, state="done" if ok else "error", progress=100,
                   error="" if ok else "files incomplete after download", cancel=False)
        logger.info("whisper install %s -> %s", name, "done" if ok else "error")
    except _Cancelled:
        _set_state(name, state="cancelled", progress=0, error="", cancel=False)
        logger.info("whisper install %s cancelled", name)
    except Exception as e:
        logger.warning("whisper install %s failed: %s", name, e)
        _set_state(name, state="error", progress=0, error=str(e)[:300], cancel=False)

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
    out += [dict(t, group="audio") for t in getattr(dm, "AUDIO_TASKS", [])]
    return out


def _task(id: str):
    for t in _catalog_tasks():
        if t["id"] == id:
            return t
    return whisper_task(id)


def _target_path(task: dict) -> Path:
    return MODELS_DIR / task["target"]


def _dir_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def is_installed(task: dict) -> bool:
    """A model is installed if it has enough data (min_gb) or is a
    single-file model with an existing file."""
    if task.get("kind") == "whisper":
        return whisper_installed(task["id"])
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
            "engine": ("url" if t.get("kind") == "url"
                       else "nemotron-asr" if group == "audio"
                       else "torch"),
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
    out += [whisper_entry(name) for name in WHISPER_SOURCES]
    return out


def _set_state(model_id: str, **kw):
    with _state_lock:
        cur = _install_state.setdefault(model_id, {"state": "idle", "progress": 0, "error": ""})
        cur.update(kw)


def _run_install(task: dict):
    if task.get("kind") == "whisper":
        _run_whisper_install(task["id"])
        return
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
    if task.get("kind") == "whisper":
        src = WHISPER_SOURCES[task["id"]]
        size = src["model_bin"]
        have = whisper_dir(task["id"]) / "model.bin"
        done = have.stat().st_size if have.is_file() else 0
        return {
            "id": model_id,
            "state": st.get("state", "done" if is_installed(task) else "idle"),
            "percent": st.get("progress", 100 if is_installed(task) else 0),
            "bytes_done": done,
            "bytes_total": size,
            "file_index": 0,
            "file_count": len(src["files"]),
            "error": st.get("error", ""),
        }
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
        "engine": ("faster-whisper" if task.get("kind") == "whisper"
                   else "url" if task.get("kind") == "url" else "torch"),
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
    task = _task(model_id)
    if task is None:
        raise KeyError(f"unknown model id: {model_id}")
    running = _install_state.get(model_id, {}).get("state") == "downloading"
    if task.get("kind") == "whisper":
        _set_state(model_id, cancel=True)
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
