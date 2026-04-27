from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive serialization helper
        return f"{type(exc).__name__}: {exc}"


def _sw_vers() -> dict[str, str]:
    try:
        raw = subprocess.run(
            ["sw_vers"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except Exception as exc:  # pragma: no cover - platform dependent
        return {"error": f"{type(exc).__name__}: {exc}"}

    payload: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def probe_torch_mps(run_tensor_test: bool = True, unsafe_force_mps: bool = False) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        return {
            "ok": False,
            "error": f"PyTorch is not installed: {exc}",
        }

    try:
        import torchvision

        torchvision_version: str | None = str(torchvision.__version__)
    except Exception:  # pragma: no cover - optional dependency
        torchvision_version = None

    backend_mps = getattr(torch.backends, "mps", None)
    backend_built = bool(_safe_call(backend_mps.is_built)) if backend_mps is not None else False
    backend_available = bool(_safe_call(backend_mps.is_available)) if backend_mps is not None else False
    mps_device_count = _safe_call(torch.mps.device_count) if hasattr(torch, "mps") else "unavailable"

    internal_has_mps = getattr(getattr(torch, "_C", None), "_has_mps", None)
    internal_mps_available = (
        _safe_call(torch._C._mps_is_available)  # type: ignore[attr-defined]
        if hasattr(getattr(torch, "_C", None), "_mps_is_available")
        else None
    )
    internal_macos_check = (
        _safe_call(torch._C._mps_is_on_macos_or_newer, 14, 0)  # type: ignore[attr-defined]
        if hasattr(getattr(torch, "_C", None), "_mps_is_on_macos_or_newer")
        else None
    )

    attempted_device = "mps" if unsafe_force_mps or backend_available else None
    tensor_test: dict[str, Any] = {
        "run_tensor_test": bool(run_tensor_test),
        "unsafe_force_mps": bool(unsafe_force_mps),
        "attempted_device": attempted_device,
    }
    if run_tensor_test:
        if attempted_device is None:
            tensor_test["skipped_reason"] = (
                "torch.backends.mps.is_available() is false; rerun with --unsafe-force-mps to attempt a real MPS tensor anyway."
            )
        else:
            try:
                tensor = torch.ones((8, 8), device=attempted_device)
                matmul = tensor @ tensor.T
                tensor_test["tensor_create_ok"] = True
                tensor_test["tensor_dtype"] = str(tensor.dtype)
                tensor_test["tensor_device"] = str(tensor.device)
                tensor_test["sum"] = float(tensor.sum().item())
                tensor_test["matmul_sum"] = float(matmul.sum().item())
                tensor_test["cpu_roundtrip_ok"] = bool(tensor.cpu().shape == tensor.shape)
            except Exception as exc:
                tensor_test["tensor_create_ok"] = False
                tensor_test["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "ok": True,
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "sw_vers": _sw_vers(),
        },
        "torch": {
            "version": str(torch.__version__),
            "torchvision_version": torchvision_version,
            "config": str(_safe_call(torch.__config__.show)),
        },
        "mps": {
            "backend_built": backend_built,
            "backend_available": backend_available,
            "device_count": mps_device_count,
            "internal_has_mps": internal_has_mps,
            "internal_mps_available": internal_mps_available,
            "internal_macos_14_plus": internal_macos_check,
        },
        "tensor_test": tensor_test,
    }
