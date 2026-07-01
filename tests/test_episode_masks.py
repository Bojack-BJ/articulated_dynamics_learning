from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_urdf_mvp.cli_parser import build_parser
from rgbd_urdf_mvp.perception.episode_masks import (
    EpisodeMaskEvaluationConfig,
    EpisodeMaskEvaluator,
    EpisodeMaskWriteConfig,
    EpisodeMaskWriter,
    _build_provider,
    _declared_part_ids,
    _filter_reference_part_ids_by_area,
    _keep_largest_component_per_label,
    _write_sam2_independent_preview_mask,
)


class EpisodeMaskWriterTest(unittest.TestCase):
    def test_mask_dir_provider_updates_episode_and_evaluates_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode_path = _write_episode(root / "episode.json")
            mask_dir = root / "provider_masks"
            mask_dir.mkdir()
            _write_mask(mask_dir / "frame_0000_mask.png", [[0, 1], [1, 1]])

            output_episode = EpisodeMaskWriter().write(
                EpisodeMaskWriteConfig(
                    episode_path=episode_path,
                    output_episode_path=root / "episode.masked.json",
                    mask_dir=mask_dir,
                    provider="mask-dir",
                    force=True,
                )
            )

            payload = json.loads(output_episode.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["frames"][0]["mask_paths_by_view"],
                ["assets/masks/mask-dir/view_0/frame_0000_mask.png"],
            )
            self.assertEqual(payload["metadata"]["mask_provider"]["provider"], "mask-dir")

            eval_path = EpisodeMaskEvaluator().evaluate(
                EpisodeMaskEvaluationConfig(
                    predicted_episode_path=output_episode,
                    reference_episode_path=output_episode,
                )
            )
            metrics = json.loads(eval_path.read_text(encoding="utf-8"))
            self.assertEqual(metrics["summary"]["mean_iou"], 1.0)

    def test_cli_parser_accepts_external_provider(self) -> None:
        args = build_parser().parse_args(
            [
                "segment-episode-masks",
                "episode.json",
                "--provider",
                "external-command",
                "--command",
                'python segment.py --image "{rgb_path}" --out "{output_mask_path}"',
                "--mask-kind",
                "part",
            ]
        )
        self.assertEqual(args.command, "segment-episode-masks")
        self.assertEqual(args.provider, "external-command")
        self.assertEqual(args.provider_command, 'python segment.py --image "{rgb_path}" --out "{output_mask_path}"')
        self.assertEqual(args.mask_kind, "part")

    def test_cli_parser_accepts_native_sam2_provider(self) -> None:
        args = build_parser().parse_args(
            [
                "segment-episode-masks",
                "episode.json",
                "--provider",
                "sam2",
                "--sam2-root",
                "sam2",
                "--sam2-checkpoint",
                "sam2/checkpoints/sam2.1_hiera_tiny.pt",
                "--sam2-device",
                "mps",
                "--sam2-prompt-mode",
                "center-box",
                "--sam2-center-box-scale",
                "0.6",
                "--sam2-mask-selection",
                "smallest",
            ]
        )
        self.assertEqual(args.provider, "sam2")
        self.assertEqual(args.sam2_device, "mps")
        self.assertEqual(args.sam2_prompt_mode, "center-box")
        self.assertEqual(args.sam2_center_box_scale, 0.6)
        self.assertEqual(args.sam2_mask_selection, "smallest")

    def test_native_sam2_provider_rejects_part_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pt"
            checkpoint.write_bytes(b"fake")
            with self.assertRaisesRegex(ValueError, "object masks only"):
                _build_provider(
                    EpisodeMaskWriteConfig(
                        episode_path=Path("episode.json"),
                        provider="sam2",
                        mask_kind="part",
                        sam2_checkpoint=checkpoint,
                    )
                )

    def test_cli_parser_accepts_segment_masks_batch(self) -> None:
        args = build_parser().parse_args(
            [
                "segment-episode-masks-batch",
                "configs/real_episodes.tsv",
                "--provider",
                "sam2",
                "--sam2-checkpoint",
                "sam2/checkpoints/sam2.1_hiera_tiny.pt",
                "--start-frame",
                "0",
                "--max-frames",
                "1",
                "--jobs",
                "2",
            ]
        )
        self.assertEqual(args.command, "segment-episode-masks-batch")
        self.assertEqual(args.max_frames, 1)
        self.assertEqual(args.jobs, 2)

    def test_cli_parser_accepts_mask_propagation(self) -> None:
        args = build_parser().parse_args(
            [
                "propagate-episode-masks",
                "outputs/real_recordings/microwave01_o/episode.sam2-first.json",
                "--backend",
                "sam2-video",
                "--sam2-checkpoint",
                "sam2/checkpoints/sam2.1_hiera_tiny.pt",
                "--reference-frame",
                "0",
                "--mask-kind",
                "part",
                "--sam2-device",
                "mps",
            ]
        )
        self.assertEqual(args.command, "propagate-episode-masks")
        self.assertEqual(args.backend, "sam2-video")
        self.assertEqual(args.sam2_device, "mps")
        self.assertEqual(args.reference_frame, 0)
        self.assertEqual(args.mask_kind, "part")

    def test_sam2_independent_preview_mask_merges_parts_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preview.png"
            _write_sam2_independent_preview_mask(
                path,
                1,
                _FakeLogits(np.asarray([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32)),
            )
            self.assertEqual(np.asarray(Image.open(path), dtype=np.uint16).tolist(), [[1, 0], [1, 0]])

            _write_sam2_independent_preview_mask(
                path,
                2,
                _FakeLogits(np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)),
            )
            self.assertEqual(np.asarray(Image.open(path), dtype=np.uint16).tolist(), [[1, 2], [1, 2]])

            _write_sam2_independent_preview_mask(
                path,
                1,
                _FakeLogits(np.asarray([[-1.0, -1.0], [1.0, -1.0]], dtype=np.float32)),
            )
            self.assertEqual(np.asarray(Image.open(path), dtype=np.uint16).tolist(), [[0, 2], [1, 2]])

    def test_sam2_preview_mask_removes_isolated_speckles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preview.png"
            _write_sam2_independent_preview_mask(
                path,
                1,
                _FakeLogits(
                    np.asarray(
                        [
                            [1.0, 1.0, -1.0],
                            [-1.0, -1.0, -1.0],
                            [-1.0, -1.0, 1.0],
                        ],
                        dtype=np.float32,
                    )
                ),
            )
            self.assertEqual(
                np.asarray(Image.open(path), dtype=np.uint16).tolist(),
                [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
            )

    def test_keep_largest_component_per_label(self) -> None:
        cleaned = _keep_largest_component_per_label(
            np.asarray(
                [
                    [1, 1, 0, 2],
                    [0, 0, 0, 2],
                    [1, 0, 2, 0],
                    [0, 0, 0, 2],
                ],
                dtype=np.uint16,
            )
        )
        self.assertEqual(cleaned.tolist(), [[1, 1, 0, 2], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]])

    def test_reference_part_filter_ignores_unknown_and_tiny_parts(self) -> None:
        metadata = {
            "part_segmentation": {
                "parts": [
                    {"part_id": 1, "name": "base"},
                    {"part_id": 2, "name": "moving"},
                ]
            }
        }
        self.assertEqual(_declared_part_ids(metadata), {1, 2})

        mask = np.zeros((480, 640), dtype=np.uint16)
        mask[:20, :20] = 1
        mask[30:50, 30:50] = 2
        mask[70:75, 70:75] = 3
        part_ids = [part_id for part_id in [1, 2, 3] if part_id in _declared_part_ids(metadata)]
        kept, ignored = _filter_reference_part_ids_by_area(mask, part_ids + [3])
        self.assertEqual(kept, [1, 2])
        self.assertEqual(ignored, [3])


def _write_episode(path: Path) -> Path:
    assets = path.parent / "assets" / "view_0"
    assets.mkdir(parents=True)
    Image.fromarray(np.full((2, 2, 3), 127, dtype=np.uint8), mode="RGB").save(assets / "frame_0000_rgb.png")
    Image.fromarray(np.ones((2, 2), dtype=np.uint16), mode="I;16").save(assets / "frame_0000_depth.png")
    payload = {
        "object_instance_id": "mask-test",
        "category": "microwave",
        "camera_intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        "frames": [
            {
                "timestamp_s": 0.0,
                "rgb_path": "assets/view_0/frame_0000_rgb.png",
                "depth_path": "assets/view_0/frame_0000_depth.png",
                "camera_pose": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "rgb_paths_by_view": ["assets/view_0/frame_0000_rgb.png"],
                "depth_paths_by_view": ["assets/view_0/frame_0000_depth.png"],
                "camera_poses_by_view": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_mask(path: Path, values: list[list[int]]) -> None:
    Image.fromarray(np.asarray(values, dtype=np.uint16), mode="I;16").save(path)


class _FakeLogits:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.array


if __name__ == "__main__":
    unittest.main()
