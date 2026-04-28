from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

from .jax_runtime import configure_jax_runtime, detect_jax_metal


def _sw_vers() -> dict[str, str] | None:
    if platform.system() != "Darwin":
        return None
    try:
        result = subprocess.run(
            ["sw_vers"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    parsed: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed or None


def probe_mjx(
    run_rollout_test: bool = True,
    *,
    jax_platform: str = "cpu",
    enable_pjrt_compatibility: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "sw_vers": _sw_vers(),
        },
        "runtime": configure_jax_runtime(
            platform=jax_platform,
            enable_pjrt_compatibility=enable_pjrt_compatibility,
        ),
        "jax_metal": detect_jax_metal(),
    }
    try:
        import jax
        import jax.numpy as jnp
    except Exception as exc:
        payload["jax"] = {
            "import_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return payload

    payload["jax"] = {
        "import_ok": True,
        "version": getattr(jax, "__version__", "unknown"),
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "local_device_count": int(jax.local_device_count()),
    }

    try:
        import mujoco
        from mujoco import mjx
    except Exception as exc:
        payload["mjx"] = {
            "import_ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return payload

    payload["mjx"] = {
        "import_ok": True,
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
    }

    if not run_rollout_test:
        payload["ok"] = True
        return payload

    xml = """
<mujoco model="mjx_probe">
  <option timestep="0.01" gravity="0 0 0" />
  <worldbody>
    <body>
      <joint type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
    </body>
  </worldbody>
</mujoco>
""".strip()

    try:
        mj_model = mujoco.MjModel.from_xml_string(xml)
        mj_data = mujoco.MjData(mj_model)
        mj_data.qvel[0] = 1.0
        jax_device = jax.devices()[0]
        # MuJoCo MJX currently does not auto-resolve Apple's METAL backend, but
        # its JAX implementation works once the device/impl pair is explicit.
        mx = mjx.put_model(mj_model, impl="jax", device=jax_device)
        dx = mjx.put_data(mj_model, mj_data, impl="jax", device=jax_device)
        stepped = jax.jit(mjx.step)(mx, dx)
        payload["rollout_test"] = {
            "ok": True,
            "qpos0": float(jnp.asarray(stepped.qpos)[0]),
            "qvel0": float(jnp.asarray(stepped.qvel)[0]),
        }
        payload["ok"] = True
    except Exception as exc:
        payload["rollout_test"] = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if (
            str(payload.get("runtime", {}).get("requested_platform", "")).lower() == "metal"
            and "default_memory_space" in str(exc)
        ):
            payload["rollout_test"]["hint"] = (
                "Metal backend initialized, but MJX rollout failed during JAX device placement. "
                "This usually indicates a jax-metal/jax compatibility issue. "
                "Use --jax-platform cpu as the safe fallback, or try a dedicated Metal environment "
                "with a known-good pinned jax/jaxlib/jax-metal stack."
            )
    return payload
