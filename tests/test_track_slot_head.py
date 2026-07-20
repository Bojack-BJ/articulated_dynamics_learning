from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rgbd_urdf_mvp.cli import build_parser
from rgbd_urdf_mvp.perception.track_slot_head import (
    TrackSlotPredictionConfig,
    TrackSlotPredictor,
    TrackSlotTrainer,
    TrackSlotTrainingConfig,
    _hungarian_slot_targets,
)


class TrackSlotHeadTests(unittest.TestCase):
    def test_hungarian_target_is_invariant_to_gt_ids(self) -> None:
        import torch

        logits = torch.tensor([[5.0, 0.0], [4.0, 0.0], [0.0, 5.0], [0.0, 4.0]])
        first = _hungarian_slot_targets(logits, torch.tensor([3, 3, 8, 8]), 2, torch)
        second = _hungarian_slot_targets(logits, torch.tensor([80, 80, 12, 12]), 2, torch)
        torch.testing.assert_close(first, second)

    def test_training_prediction_and_default_viewer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            for object_index, split in enumerate(("train", "val")):
                track_path = root / f"object{object_index}.json"
                feature_path = root / f"object{object_index}.npz"
                tracks = []
                features = []
                ids = []
                for index in range(12):
                    label = 10 if index < 6 else 20
                    track_id = object_index * 100 + index
                    vector = np.asarray([1.0, 0.0] if label == 10 else [0.0, 1.0], dtype=np.float32)
                    vector += np.random.default_rng(index).normal(0, 0.01, 2).astype(np.float32)
                    vector /= np.linalg.norm(vector)
                    ids.append(track_id)
                    features.append(vector)
                    tracks.append({
                        "track_id": track_id, "original_part_id": label, "part_id": 1,
                        "samples": [{"frame_index": 0, "timestamp_s": 0.0, "visible": True, "xyz_world": [float(index), 0.0, 0.0]}],
                    })
                track_path.write_text(json.dumps({"tracks": tracks}), encoding="utf-8")
                np.savez_compressed(feature_path, track_ids=np.asarray(ids), embeddings=np.stack(features))
                rows.append(f"object{object_index}\t{track_path}\t{feature_path}\t{split}")
            manifest = root / "manifest.tsv"
            manifest.write_text("object_id\ttracks_path\tfeatures_npz\tsplit\n" + "\n".join(rows) + "\n", encoding="utf-8")
            model = TrackSlotTrainer(TrackSlotTrainingConfig(
                manifest_path=manifest, output_dir=root / "model", max_slots=2,
                hidden_dim=16, epochs=20, learning_rate=0.01, device="cpu",
            )).train()
            output, viewer = TrackSlotPredictor(TrackSlotPredictionConfig(
                tracks_path=root / "object1.json", features_npz=root / "object1.npz",
                model_path=model, output_json=root / "predicted.json", device="cpu",
            )).predict()
            payload = json.loads(output.read_text(encoding="utf-8"))
            predicted = [track["part_id"] for track in payload["tracks"]]
            self.assertEqual(len(set(predicted[:6])), 1)
            self.assertEqual(len(set(predicted[6:])), 1)
            self.assertNotEqual(predicted[0], predicted[-1])
            self.assertTrue(viewer and viewer.exists())

    def test_cli(self) -> None:
        parser = build_parser()
        train = parser.parse_args(["train-track-slot-head", "manifest.tsv", "--output-dir", "model"])
        predict = parser.parse_args([
            "predict-track-slots", "tracks.json", "features.npz", "model.pt",
            "--output-json", "predicted.json",
        ])
        self.assertEqual(train.command, "train-track-slot-head")
        self.assertEqual(predict.command, "predict-track-slots")


if __name__ == "__main__":
    unittest.main()
