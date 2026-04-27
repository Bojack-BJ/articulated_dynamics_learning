from __future__ import annotations

import unittest

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.core.mjx_probe import probe_mjx


class MJXProbeTests(unittest.TestCase):
    def test_probe_mjx_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["probe-mjx", "--no-rollout-test"])
        self.assertEqual(args.command, "probe-mjx")
        self.assertTrue(args.no_rollout_test)

    def test_probe_mjx_returns_structured_payload(self) -> None:
        payload = probe_mjx(run_rollout_test=False)
        self.assertIn("ok", payload)
        self.assertIn("system", payload)
        self.assertIn("jax", payload)


if __name__ == "__main__":
    unittest.main()
