from __future__ import annotations

import unittest

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.core.mjx_probe import probe_mjx


class MJXProbeTests(unittest.TestCase):
    def test_probe_mjx_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["probe-mjx", "--jax-platform", "cpu", "--enable-pjrt-compatibility", "--no-rollout-test"]
        )
        self.assertEqual(args.command, "probe-mjx")
        self.assertEqual(args.jax_platform, "cpu")
        self.assertTrue(args.enable_pjrt_compatibility)
        self.assertTrue(args.no_rollout_test)

    def test_probe_mjx_returns_structured_payload(self) -> None:
        payload = probe_mjx(run_rollout_test=False, jax_platform="cpu")
        self.assertIn("ok", payload)
        self.assertIn("system", payload)
        self.assertIn("jax", payload)
        self.assertIn("runtime", payload)


if __name__ == "__main__":
    unittest.main()
