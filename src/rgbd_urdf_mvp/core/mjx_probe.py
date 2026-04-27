from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any


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


def probe_mjx(run_rollout_test: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "sw_vers": _sw_vers(),
        },
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
        mx = mjx.put_model(mj_model)
        dx = mjx.put_data(mj_model, mj_data)
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
    return payload
