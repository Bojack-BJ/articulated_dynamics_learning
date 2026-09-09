from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_neural_head_so3_pilot import _selected_variants, prepare_manifest


class NeuralHeadSo3PilotTests(unittest.TestCase):
    def test_selected_variants_supports_focused_reruns(self) -> None:
        arguments = argparse.Namespace(variants="direct_no_aug, vector_neuron")
        self.assertEqual(
            _selected_variants(arguments),
            ["direct_no_aug", "vector_neuron"],
        )

    def test_selected_variants_rejects_unknown_names(self) -> None:
        arguments = argparse.Namespace(variants="direct_no_aug,unknown")
        with self.assertRaisesRegex(ValueError, "Unknown pilot variants"):
            _selected_variants(arguments)

    def test_prepare_manifest_is_deterministic_and_has_fixed_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tsv"
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("object_id", "tracks_path", "features_npz", "split"),
                    delimiter="\t",
                )
                writer.writeheader()
                for index in range(50):
                    writer.writerow({
                        "object_id": f"category{index % 5}_{index:03d}",
                        "tracks_path": root / f"missing_{index}.json",
                        "features_npz": root / f"missing_{index}.npz",
                        "split": "train",
                    })
            arguments = argparse.Namespace(
                manifest=source,
                output_dir=root / "output",
                train_count=24,
                val_count=8,
                test_count=12,
            )
            first = prepare_manifest(arguments).read_text(encoding="utf-8")
            second = prepare_manifest(arguments).read_text(encoding="utf-8")
            self.assertEqual(first, second)
            rows = list(csv.DictReader(first.splitlines(), delimiter="\t"))
            self.assertEqual(sum(row["split"] == "train" for row in rows), 24)
            self.assertEqual(sum(row["split"] == "val" for row in rows), 8)
            self.assertEqual(sum(row["split"] == "test" for row in rows), 12)

    def test_prepare_manifest_uses_catalog_strata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.tsv"
            objects = []
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("object_id", "tracks_path", "features_npz", "split"),
                    delimiter="\t",
                )
                writer.writeheader()
                for index in range(6):
                    object_id = f"partnet_{index}"
                    writer.writerow({
                        "object_id": object_id,
                        "tracks_path": root / f"missing_{index}.json",
                        "features_npz": root / f"missing_{index}.npz",
                        "split": "train",
                    })
                    objects.append({
                        "object_id": object_id,
                        "category": "drawer" if index < 3 else "refrigerator",
                        "joint_count": 1 if index % 2 == 0 else 4,
                        "joint_types": ["prismatic"] if index < 3 else ["revolute"],
                    })
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({"objects": objects}), encoding="utf-8")
            arguments = argparse.Namespace(
                manifest=source,
                catalog=catalog,
                output_dir=root / "output",
                train_count=2,
                val_count=2,
                test_count=2,
            )
            prepare_manifest(arguments)
            selection = json.loads(
                (root / "output" / "pilot_selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {row["category"] for row in selection["objects"]},
                {"drawer", "refrigerator"},
            )
            self.assertEqual(
                {row["complexity"] for row in selection["objects"]}, {"2", ">=5"}
            )


if __name__ == "__main__":
    unittest.main()
