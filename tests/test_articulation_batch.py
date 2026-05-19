from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.batch.articulation_pipeline import (
    UsdMjcfBatchConfig,
    UsdMjcfBatchConverter,
    build_cli_argv_from_template,
    infer_joint_name,
    mjcf_output_prefix_for_usd,
    parse_batch_manifest,
)


class ArticulationBatchTests(unittest.TestCase):
    def test_parse_batch_manifest_skips_comments_and_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "batch.tsv"
            manifest_path.write_text(
                "\n".join(
                    [
                        "# category\tmodel_path\tobject_id\tjoint_name",
                        "",
                        "Microwave\texamples/mujoco_models/Microwave011.xml\tmw011\tauto",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            specs = parse_batch_manifest(manifest_path)

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].category, "microwave")
            self.assertEqual(specs[0].object_id, "mw011")

    def test_infer_joint_name_prefers_category_specific_token(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_xml = Path(temp_dir) / "model.xml"
            model_xml.write_text(
                "\n".join(
                    [
                        "<mujoco>",
                        "  <worldbody>",
                        "    <body name='root'>",
                        "      <joint name='aux_hinge' type='hinge'/>",
                        "      <joint name='microjoint' type='hinge'/>",
                        "    </body>",
                        "  </worldbody>",
                        "</mujoco>",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            inferred = infer_joint_name("microwave", model_xml, "auto")

            self.assertEqual(inferred, "microjoint")

    def test_build_cli_argv_from_template_merges_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "track.yaml"
            output_json = Path(temp_dir) / "tracks.json"
            config_path.write_text(
                "\n".join(
                    [
                        "command: track-part-pixels",
                        "args:",
                        "  device: auto",
                        "  frame-stride: 4",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = build_cli_argv_from_template(
                config_path,
                {
                    "episode": "episode.json",
                    "output-json": output_json,
                    "device": "mps",
                },
            )

            self.assertEqual(argv[0], "track-part-pixels")
            self.assertEqual(argv[1], "episode.json")
            self.assertIn("--output-json", argv)
            self.assertIn(str(output_json), argv)
            self.assertIn("--frame-stride", argv)
            self.assertIn("4", argv)
            device_index = argv.index("--device")
            self.assertEqual(argv[device_index + 1], "mps")

    def test_run_articulation_batch_parser_accepts_dynamics_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-articulation-batch",
                "configs/batch_objects_example.tsv",
                "--dynamics-backend",
                "mjx",
                "--dynamics-jax-platform",
                "metal",
                "--dynamics-enable-pjrt-compatibility",
                "--dynamics-render-gl-backend",
                "cgl",
                "--plot-dynamics",
            ]
        )

        self.assertEqual(args.command, "run-articulation-batch")
        self.assertEqual(args.dynamics_backend, "mjx")
        self.assertEqual(args.dynamics_jax_platform, "metal")
        self.assertTrue(args.dynamics_enable_pjrt_compatibility)
        self.assertEqual(args.dynamics_render_gl_backend, "cgl")
        self.assertTrue(args.plot_dynamics)

    def test_convert_usd_mjcf_batch_parser_accepts_arguments(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "convert-usd-mjcf-batch",
                "configs/batch_objects_example.tsv",
                "--output-dir",
                "outputs/converted_mjcf",
                "--converted-manifest",
                "configs/batch_converted.tsv",
                "--jobs",
                "2",
                "--force",
            ]
        )

        self.assertEqual(args.command, "convert-usd-mjcf-batch")
        self.assertEqual(str(args.output_dir), "outputs/converted_mjcf")
        self.assertEqual(str(args.converted_manifest), "configs/batch_converted.tsv")
        self.assertEqual(args.jobs, 2)
        self.assertTrue(args.force)

    def test_usd_batch_converter_skips_existing_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            usd_path = temp_path / "Cabinet001.usd"
            usd_path.write_text("#usda 1.0\n", encoding="utf-8")
            output_dir = temp_path / "mjcf"
            output_dir.mkdir()
            existing_xml = mjcf_output_prefix_for_usd(usd_path, output_dir).with_suffix(".xml")
            existing_xml.write_text("<mujoco/>\n", encoding="utf-8")
            manifest_path = temp_path / "batch.tsv"
            manifest_path.write_text(
                f"drawer\t{usd_path}\tcabinet001\tauto\n",
                encoding="utf-8",
            )
            converted_manifest = temp_path / "converted.tsv"

            result = UsdMjcfBatchConverter(
                UsdMjcfBatchConfig(
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                    converted_manifest=converted_manifest,
                )
            ).run()

            self.assertEqual(result["converted"], 0)
            self.assertEqual(result["skipped_existing"], 1)
            self.assertTrue(converted_manifest.exists())
            self.assertIn(str(existing_xml), converted_manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
