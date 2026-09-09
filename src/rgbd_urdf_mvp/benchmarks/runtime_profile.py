"""Runtime and hardware metadata shared by benchmark entry points."""

from __future__ import annotations

import os
import platform
import socket
from typing import Any


def hardware_snapshot(torch: Any | None = None) -> dict[str, Any]:
    """Return paper-reportable execution metadata without requiring CUDA."""
    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_logical_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }
    if torch is None:
        return payload
    cuda_available = bool(torch.cuda.is_available())
    payload["torch"] = {
        "version": str(torch.__version__),
        "num_threads": int(torch.get_num_threads()),
        "num_interop_threads": int(torch.get_num_interop_threads()),
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": (
            int(torch.backends.cudnn.version())
            if cuda_available and torch.backends.cudnn.is_available()
            else None
        ),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    payload["gpus"] = [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(index).total_memory
            ),
        }
        for index in range(torch.cuda.device_count())
    ]
    return payload


def synchronize_device(torch: Any, device: str) -> None:
    """Synchronize asynchronous accelerators before stopping a timer."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif str(device).startswith("mps") and hasattr(torch, "mps"):
        torch.mps.synchronize()
