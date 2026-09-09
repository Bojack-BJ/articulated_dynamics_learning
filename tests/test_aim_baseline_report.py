from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.aim_baseline_report import (
    complexity_bucket,
    write_extended_report,
)


def _evaluation(path: Path, *, coverage: float, iou: float, ari: float, pred: int, gt: int) -> None:
    metrics = {
        "undersegmented_gt_part_count": max(gt - pred, 0),
        "unmatched_gt_part_count": max(gt - pred, 0),
        "largest_cluster_ratio": 0.6,
        "adjusted_rand_index": ari,
    }
    path.write_text(json.dumps({
        "predicted_part_count": pred,
        "gt_part_count": gt,
        "primary": {
            "distance_ratio_bbox": 1.0,
            "covered_only_metrics": metrics,
            "coverage_aware_metrics": {"one_to_one_mean_iou": iou},
        },
        "thresholds": [{
            "distance_ratio_bbox": 0.02,
            "geometry_coverage": coverage,
        }],
    }), encoding="utf-8")


class AimBaselineReportTest(unittest.TestCase):
    def test_complexity_buckets(self) -> None:
        self.assertEqual(complexity_bucket(2), "2")
        self.assertEqual(complexity_bucket(4), "3-4")
        self.assertEqual(complexity_bucket(5), ">=5")

    def test_writes_paired_and_failure_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _evaluation(root / "shared.json", coverage=0.4, iou=0.2, ari=0.1, pred=2, gt=5)
            _evaluation(root / "style.json", coverage=0.9, iou=0.15, ari=0.05, pred=2, gt=5)
            manifest = root / "runs.csv"
            fields = [
                "object_id", "category", "method", "protocol", "status",
                "evaluation_json", "failure_stage", "exception",
            ]
            with manifest.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {
                        "object_id": "partnet_1", "category": "coffeemachine",
                        "method": "aim", "protocol": "shared_3view", "status": "success",
                        "evaluation_json": "shared.json",
                    },
                    {
                        "object_id": "partnet_1", "category": "coffeemachine",
                        "method": "aim", "protocol": "aim_style", "status": "success",
                        "evaluation_json": "style.json",
                    },
                    {
                        "object_id": "partnet_2", "category": "oven",
                        "method": "aim", "protocol": "shared_3view", "status": "failed",
                        "failure_stage": "segmentation", "exception": "RANSAC failed",
                    },
                ])
            output = root / "report"
            result = write_extended_report(manifest, output)
            self.assertEqual(
                result["protocol_pair_aggregates"][0]["delta_coverage_mean"], 0.5
            )
            failures = json.loads((output / "failures.json").read_text(encoding="utf-8"))
            self.assertEqual(failures["failure_reason_histogram"], {"segmentation": 1})
            pairs = list(csv.DictReader((output / "protocol_pairs.csv").open(encoding="utf-8")))
            self.assertEqual(len(pairs), 1)
            summary = (output / "summary.md").read_text(encoding="utf-8").lower()
            self.assertIn("high-coverage", summary)
            self.assertIn("aim-style approximation", summary)


if __name__ == "__main__":
    unittest.main()
