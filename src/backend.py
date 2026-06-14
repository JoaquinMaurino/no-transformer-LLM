import os
from typing import Any, Optional, Tuple

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:
    cp = None

_xp = np
_backend_name = "numpy"


def _env_device() -> str:
    # Priority: explicit device selector, then legacy USE_GPU flag.
    raw = os.getenv("WORD2VEC_DEVICE")
    if raw:
        return raw.strip().lower()

    use_gpu = os.getenv("USE_GPU")
    if use_gpu is None:
        return "auto"

    value = use_gpu.strip().lower()
    if value in {"1", "true", "yes", "on", "gpu"}:
        return "gpu"
    if value in {"0", "false", "no", "off", "cpu"}:
        return "cpu"
    return "auto"


def configure_backend(device: Optional[str] = None) -> str:
    global _xp, _backend_name

    requested = (device or _env_device()).strip().lower()

    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("device debe ser uno de: auto, cpu, gpu")

    if requested == "cpu":
        _xp = np
        _backend_name = "numpy"
        return _backend_name

    if requested == "gpu":
        if cp is None:
            raise RuntimeError(
                "Se solicito GPU pero CuPy no esta instalado. "
                "Instala una build de CuPy compatible con tu CUDA."
            )
        _xp = cp
        _backend_name = "cupy"
        return _backend_name

    # auto
    if cp is not None:
        try:
            device_count = int(cp.cuda.runtime.getDeviceCount())
            if device_count > 0:
                _xp = cp
                _backend_name = "cupy"
                return _backend_name
        except Exception:
            pass

    _xp = np
    _backend_name = "numpy"
    return _backend_name


def xp():
    return _xp


def backend_name() -> str:
    return _backend_name


def is_gpu_backend() -> bool:
    return _backend_name == "cupy"


def set_seed(seed: int) -> None:
    _xp.random.seed(seed)


def random_permutation(n: int):
    return _xp.random.permutation(n)


def zeros(shape, dtype=None):
    return _xp.zeros(shape, dtype=dtype)


def arange(n: int):
    return _xp.arange(n)


def as_float32(x):
    return _xp.asarray(x, dtype=_xp.float32)


def as_int32(x):
    return _xp.asarray(x, dtype=_xp.int32)


def to_backend(x, dtype=None):
    if is_gpu_backend():
        return cp.asarray(x, dtype=dtype)
    return np.asarray(x, dtype=dtype)


def to_cpu(x):
    if is_gpu_backend():
        if isinstance(x, np.ndarray):
            return x
        return cp.asnumpy(x)
    return np.asarray(x)


def to_float(x) -> float:
    if isinstance(x, (float, int)):
        return float(x)

    if is_gpu_backend():
        return float(cp.asnumpy(x).item())

    if hasattr(x, "item"):
        return float(x.item())

    return float(x)


def contains_nan(x) -> bool:
    return bool(to_float(_xp.any(_xp.isnan(x))))


def scatter_add_rows(dest, row_indices, values) -> None:
    _xp.add.at(dest, row_indices, values)


def gpu_memory_info_mb() -> Optional[Tuple[float, float]]:
    if not is_gpu_backend():
        return None

    try:
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        used_mb = (total_bytes - free_bytes) / (1024.0 * 1024.0)
        total_mb = total_bytes / (1024.0 * 1024.0)
        return used_mb, total_mb
    except Exception:
        return None


configure_backend(None)
