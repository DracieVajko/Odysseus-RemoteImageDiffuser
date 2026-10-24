"""Shared VRAM/RAM gate — at most one heavy model resident.

SD and Whisper compete for the same VRAM (6 GB card, 3-4 GB limit on mobile).
Switching kind/id = unload + load. Cold load returns the 503 loading shape.
"""
import threading
import time


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
        self.last_used = 0.0
        self._in_flight = 0
        # Called on unload (unified_server hooks engine release).
        self.on_unload = None

    def set_loading(self, kind, model_id, active):
        with self._lock:
            self._loading = bool(active)
            self._loading_model = model_id if active else ""

    def ensure(self, kind, model_id):
        # The release callback runs while the lock is held, so a concurrent
        # `ensure` cannot install a new model before the old engine is released.
        with self._lock:
            if self._loading:
                raise ModelLoadingError(
                    detail="loading model",
                    loading_model=self._loading_model or model_id,
                    retry_after=30,
                )
            if self._loaded_kind == kind and self._loaded_id == model_id:
                self.last_used = time.time()
                return model_id
            # switch = unload the previous one (and release its engine)
            if self._loaded_kind is not None:
                self._do_unload_locked("switch")
                if callable(self.on_unload):
                    try:
                        self.on_unload("switch")
                    except Exception:
                        pass
            self._loaded_kind = kind
            self._loaded_id = model_id
            self.load_count += 1
            self.last_used = time.time()
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
            had = self._do_unload_locked(reason)
            if had and callable(self.on_unload):
                try:
                    self.on_unload(reason)
                except Exception:
                    pass
            return had

    def _refresh_busy_locked(self):
        """busy = a request runs, queue = requests waiting behind the first one."""
        self._busy = self._in_flight > 0
        self._queue = max(0, self._in_flight - 1)

    def begin(self):
        """Mark an in-flight request — the idle unloader must not unload during it."""
        with self._lock:
            self._in_flight += 1
            self._refresh_busy_locked()
            self.last_used = time.time()

    def end(self):
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._refresh_busy_locked()
            self.last_used = time.time()

    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def idle_seconds(self) -> float:
        with self._lock:
            if self._loaded_kind is None or self.last_used <= 0:
                return 0.0
            return max(0.0, time.time() - self.last_used)

    def status(self):
        with self._lock:
            return {
                "loaded_kind": self._loaded_kind,
                "loaded_id": self._loaded_id,
                "loading": self._loading,
                "loading_model": self._loading_model,
                "busy": self._busy,
                "queue": self._queue,
                "in_flight": self._in_flight,
                "idle_seconds": (round(time.time() - self.last_used, 1)
                                 if self._loaded_kind is not None and self.last_used > 0 else None),
            }
