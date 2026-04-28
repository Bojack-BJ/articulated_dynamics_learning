from __future__ import annotations

import os
import unittest

from rgbd_urdf_mvp.core.jax_runtime import configure_jax_runtime


class JAXRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "ENABLE_PJRT_COMPATIBILITY": os.environ.get("ENABLE_PJRT_COMPATIBILITY"),
        }

    def tearDown(self) -> None:
        for key, value in self._previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_configure_jax_runtime_defaults_to_cpu(self) -> None:
        payload = configure_jax_runtime(platform="cpu", enable_pjrt_compatibility=None)
        self.assertEqual(payload["requested_platform"], "cpu")
        self.assertEqual(os.environ.get("JAX_PLATFORMS"), "cpu")

    def test_configure_jax_runtime_enables_pjrt_for_metal_by_default(self) -> None:
        payload = configure_jax_runtime(platform="metal", enable_pjrt_compatibility=None)
        self.assertEqual(payload["requested_platform"], "metal")
        self.assertEqual(os.environ.get("JAX_PLATFORMS"), "METAL,cpu")
        self.assertEqual(os.environ.get("ENABLE_PJRT_COMPATIBILITY"), "1")


if __name__ == "__main__":
    unittest.main()
