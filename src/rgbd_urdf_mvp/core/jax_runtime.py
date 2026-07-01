from __future__ import annotations

import importlib.metadata
import os
from typing import Any


def configure_jax_runtime(
    *,
    platform: str = "cpu",
    enable_pjrt_compatibility: bool | None = None,
) -> dict[str, Any]:
    """Apply JAX backend env vars before importing jax.

    The default stays on CPU even when `jax-metal` is installed. This avoids
    surprising crashes in headless or sandboxed environments that cannot see
    the local Apple GPU.
    """

    normalized = str(platform or "cpu").strip().lower()
    if normalized not in {"cpu", "metal", "auto"}:
        raise ValueError(f"Unsupported JAX platform: {platform!r}")

    if enable_pjrt_compatibility is None:
        enable_pjrt_compatibility = normalized == "metal"

    env_updates: dict[str, str] = {}
    if normalized == "cpu":
        env_updates["JAX_PLATFORMS"] = "cpu"
    elif normalized == "metal":
        # Apple's Metal plugin registers itself as the uppercase PJRT backend.
        env_updates["JAX_PLATFORMS"] = "METAL,cpu"
    else:
        os.environ.pop("JAX_PLATFORMS", None)

    if enable_pjrt_compatibility:
        env_updates["ENABLE_PJRT_COMPATIBILITY"] = "1"

    for key, value in env_updates.items():
        os.environ[key] = value

    return {
        "requested_platform": normalized,
        "enable_pjrt_compatibility": bool(enable_pjrt_compatibility),
        "env": {key: os.environ.get(key) for key in ("JAX_PLATFORMS", "ENABLE_PJRT_COMPATIBILITY")},
    }


def detect_jax_metal() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("jax-metal")
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False, "version": None}
    return {"installed": True, "version": version}
