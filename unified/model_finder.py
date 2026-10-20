"""Model finder (HuggingFace search) + custom model add + delete.

Bezpečnostné pravidlo: add/delete pracujú výhradne pod models/.
Delete odmietne zmazať katalógový model (na to je /v1/models/uninstall),
aby náhodný klik neodstránil 19 GB základných váh.
"""
import logging
import shutil
import threading
import urllib.parse
import urllib.request
from pathlib import Path

logger = logging.getLogger("unified.finder")

HERE = Path(__file__).resolve().parent.parent
MODELS_DIR = HERE / "models"

_search_state: dict = {}


def _hf_get(url: str, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "odysseus-unified"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        import json as _json
        return _json.loads(r.read().decode())


def search_hf(query: str, limit: int = 20, filter_task: str = "") -> dict:
    """Hľadá modely na HuggingFace (server-side, bez CORS problémov)."""
    q = (query or "").strip()
    params = {"search": q, "limit": str(max(1, min(50, limit))), "full": "false"}
    if filter_task:
        params["filter"] = filter_task
    url = "https://huggingface.co/api/models?" + urllib.parse.urlencode(params)
    try:
        data = _hf_get(url)
    except Exception as e:
        raise RuntimeError(f"HF search failed: {e}")
    out = []
    for m in data if isinstance(data, list) else []:
        out.append({
            "id": m.get("id") or m.get("modelId", ""),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "pipeline_tag": m.get("pipeline_tag", ""),
            "library": m.get("library_name", ""),
            "tags": (m.get("tags") or [])[:8],
            "gated": bool(m.get("gated")),
        })
    return {"query": q, "count": len(out), "models": out}


def _catalog_ids() -> set:
    try:
        import download_models as dm
        return {t["id"] for t in (dm.TASKS + dm.UPSCALE_TASKS + dm.VLM_TASKS)}
    except Exception:
        return set()


def _custom_dirs() -> list:
    """Adresáre v models/, ktoré nie sú registrované v models.json ani katalógu."""
    import json as _json
    known = set()
    try:
        cfg = _json.loads((HERE / "models.json").read_text(encoding="utf-8"))
        for prof in (cfg.get("models") or {}).values():
            p = prof.get("path") or prof.get("file")
            if p:
                known.add(Path(p).parts[-2] if Path(p).is_file() else Path(p).name)
    except Exception:
        pass
    # Katalógové modely (download_models.py) tiež nie sú "custom".
    try:
        import download_models as dm
        for t in (dm.TASKS + dm.UPSCALE_TASKS + dm.VLM_TASKS):
            if t.get("target"):
                known.add(t["target"])
    except Exception:
        pass
    out = []
    if not MODELS_DIR.is_dir():
        return out
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name in known:
            continue
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        out.append({"name": d.name, "path": str(d), "size_gb": round(size / 1e9, 2)})
    return out


def custom_models() -> dict:
    return {"models": _custom_dirs(), "models_dir": str(MODELS_DIR)}


def start_add(repo_id: str, name: str = "") -> dict:
    rid = (repo_id or "").strip()
    if "/" not in rid:
        raise ValueError("repo_id must look like 'owner/name'")
    target_name = (name or rid.split("/")[-1]).replace("..", "").replace("/", "_").strip()
    if not target_name:
        raise ValueError("empty model name")
    key = f"custom:{target_name}"
    st = _search_state.get(key, {})
    if st.get("state") == "downloading":
        return {"id": target_name, "state": "downloading", "already_running": True}
    _search_state[key] = {"state": "downloading", "progress": 1, "error": ""}
    threading.Thread(target=_run_add, args=(rid, target_name, key), daemon=True).start()
    return {"id": target_name, "repo_id": rid, "state": "downloading"}


def _run_add(repo_id: str, target_name: str, key: str):
    try:
        from huggingface_hub import snapshot_download
        dest = MODELS_DIR / target_name
        dest.mkdir(parents=True, exist_ok=True)
        allow = ["*.json", "*.safetensors", "*.txt", "*.model", "*.bin",
                 "tokenizer/*", "scheduler/*", "text_encoder/*", "text_encoder_2/*",
                 "vae/*", "unet/*", "transformer/*", "feature_extractor/*",
                 "preprocessor_config.json", "*.jinja"]
        logger.info("adding custom model %s -> %s", repo_id, dest)
        snapshot_download(repo_id, local_dir=str(dest), allow_patterns=allow)
        _search_state[key] = {"state": "done", "progress": 100, "error": ""}
        logger.info("custom model %s ready", target_name)
    except Exception as e:
        logger.warning("add %s failed: %s", repo_id, e)
        _search_state[key] = {"state": "error", "progress": 0, "error": str(e)[:300]}


def add_state(name: str) -> dict:
    return _search_state.get(f"custom:{name}", {"state": "idle", "progress": 0, "error": ""})


def delete_custom(name: str) -> dict:
    """Zmaže models/<name>. Odmietne katalógové modely a cesty mimo models/."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError("invalid model name")
    if name in _catalog_ids():
        raise ValueError("catalog model: use /v1/models/uninstall instead")
    target = (MODELS_DIR / name).resolve()
    guard = MODELS_DIR.resolve()
    if guard not in target.parents:
        raise ValueError("refusing to delete outside models/")
    if not target.is_dir():
        raise KeyError(f"model directory not found: {name}")
    shutil.rmtree(target)
    _search_state.pop(f"custom:{name}", None)
    logger.info("deleted custom model %s", target)
    return {"name": name, "removed": not target.exists()}
