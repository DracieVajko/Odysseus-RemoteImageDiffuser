"""ONNX Runtime environment helpers.

Two jobs:
1. Register the NVIDIA DLL directories installed by the `nvidia-*` pip
   packages (cuDNN/cuBLAS) so onnxruntime-gpu can find them on Windows.
2. Pick execution providers per model kind:
   - `qdq`  : INT8 QDQ models. DirectML cannot fuse these graphs, so DML is
              skipped; CUDA (if available) or CPU is used.
   - `generic`: FP32/FP16 models. CUDA first, then DirectML, then CPU.

Override with `PRISM_ONNX_PROVIDER=cuda|dml|cpu` (default: auto).
"""
import os
import sys
from pathlib import Path

_REGISTERED = False


def register_nvidia_dlls() -> list:
    """Add pip-installed NVIDIA DLL directories to the Windows search path."""
    global _REGISTERED
    dirs = []
    if sys.platform != "win32":
        return dirs
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if spec and spec.submodule_search_locations:
            base = Path(list(spec.submodule_search_locations)[0])
            for d in base.glob("*/bin"):
                dirs.append(str(d))
            for d in base.glob("*/bin/x86_64"):
                dirs.append(str(d))
    except Exception:
        pass
    # Also cover a standard CUDA toolkit install (cuDNN copied there).
    for env in ("CUDA_PATH",):
        root = os.environ.get(env)
        if root:
            bin_dir = Path(root) / "bin"
            if bin_dir.is_dir():
                dirs.append(str(bin_dir))
    for d in dirs:
        try:
            os.add_dll_directory(d)
        except Exception:
            pass
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    _REGISTERED = bool(dirs)
    return dirs


def available_providers() -> list:
    register_nvidia_dlls()
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


_CUDA_OPTS = {
    # HEURISTIC avoids the long cuDNN algorithm benchmark on the first run.
    # max_workspace=0 + kSameAsRequested keep the 6 GB card from thrashing
    # (default arena growth pushed 512x512 into shared memory and took 5+ min).
    "cudnn_conv_algo_search": "HEURISTIC",
    "cudnn_conv_use_max_workspace": "0",
    "arena_extend_strategy": "kSameAsRequested",
}
_CUDA = ("CUDAExecutionProvider", _CUDA_OPTS)


def pick_providers(kind: str = "generic") -> list:
    """Return the provider list for a model kind (`qdq` or `generic`)."""
    override = (os.environ.get("PRISM_ONNX_PROVIDER") or "auto").strip().lower()
    avail = available_providers()

    if override == "cpu":
        return ["CPUExecutionProvider"]
    if override == "cuda":
        return ([_CUDA, "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in avail else ["CPUExecutionProvider"])
    if override == "dml":
        if kind == "qdq":
            return ["CPUExecutionProvider"]
        return (["DmlExecutionProvider", "CPUExecutionProvider"]
                if "DmlExecutionProvider" in avail else ["CPUExecutionProvider"])

    # auto
    if "CUDAExecutionProvider" in avail:
        return [_CUDA, "CPUExecutionProvider"]
    if kind != "qdq" and "DmlExecutionProvider" in avail:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def provider_label(providers: list) -> str:
    if not providers:
        return "unknown"
    p = providers[0]
    if isinstance(p, (tuple, list)):
        p = p[0]
    return {
        "CUDAExecutionProvider": "cuda",
        "DmlExecutionProvider": "directml",
        "CPUExecutionProvider": "cpu",
    }.get(p, p)
