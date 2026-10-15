"""Shared VRAM/RAM gate — vždy max 1 ťažký model rezidentný.

SD aj Whisper súťažia o rovnakú VRAM (6 GB karta, 3-4 GB limit na mobile).
Prepnutie druhu/id = unload + load. Cold load hlási 503 loading tvar.
"""
import threading


class ModelLoadingError(Exception):
    def __init__(self, detail="loading model", loading_model="", retry_after=30):
        super().__init__(detail)
        self.detail = detail
        self.loading_model = loading_model
        self.retry_after = retry_after


class ModelBusyError(Exception):
    pass


class ModelManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._loaded_kind = None
        self._loaded_id = None
        self._loading = False
        self._loading_model = ""
        self._busy = False
        self._queue = 0
        self.load_count = 0

    def set_loading(self, kind, model_id, active):
        with self._lock:
            self._loading = bool(active)
            self._loading_model = model_id if active else ""

    def ensure(self, kind, model_id):
        with self._lock:
            if self._loading:
                raise ModelLoadingError(
                    detail="loading model",
                    loading_model=self._loading_model or model_id,
                    retry_after=30,
                )
            if self._loaded_kind == kind and self._loaded_id == model_id:
                return model_id
            # prepnutie = unload predchádzajúceho
            if self._loaded_kind is not None:
                self._do_unload_locked("switch")
            self._loaded_kind = kind
            self._loaded_id = model_id
            self.load_count += 1
            return model_id

    def _do_unload_locked(self, reason=""):
        had = self._loaded_kind is not None
        self._loaded_kind = None
        self._loaded_id = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return had

    def unload(self, reason="idle"):
        with self._lock:
            return self._do_unload_locked(reason)

    def status(self):
        with self._lock:
            return {
                "loaded_kind": self._loaded_kind,
                "loaded_id": self._loaded_id,
                "loading": self._loading,
                "loading_model": self._loading_model,
                "busy": self._busy,
                "queue": self._queue,
            }
