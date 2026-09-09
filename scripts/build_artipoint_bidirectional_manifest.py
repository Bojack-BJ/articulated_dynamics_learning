#!/usr/bin/env python3
"""Build the protocol manifest for the ArtiPoint bidirectional comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PARTNET_METHODS = (
    "track2art_hybrid",
    "artipoint_object_mask_adapter",
    "aim",
    "reart",
    "dta",
    "artgs",
    "videoartgs",
    "gaussianart",
    "paris",
    "ditto",
)

ARTI4D_METHODS = (
    "artipoint_native",
    "track2art_object_mask_adapter",
    "aim_applicable_adapter",
    "reart_applicable_adapter",
    "dta_unsupported_input",
    "artgs_unsupported_input",
    "videoartgs_applicable_adapter",
    "gaussianart_unsupported_input",
    "paris_unsupported_input",
    "ditto_unsupported_input",
)


def manifest_rows() -> list[dict[str, object]]:
    return [
        {
            "dataset": "partnet_mobility",
            "object_id": "partnet_10944",
            "category": "refrigerator",
            "interaction": "single_revolute",
            "gt_part_count": 2,
            "protocol": "partnet_rgbd_interaction",
            "methods": ";".join(PARTNET_METHODS),
            "gt_interaction_cue": False,
            "gt_part_labels_in_inference": False,
            "notes": "Aligned object with existing all-baseline results; ArtiPoint uses an object-mask cue adapter because no hand is rendered.",
        },
        {
            "dataset": "arti4d_rh078",
            "object_id": "scene_2025-04-09-10-38-38",
            "category": "drawer",
            "interaction": "right-drawer-1",
            "gt_part_count": 2,
            "protocol": "official_arti4d_interaction_window",
            "methods": ";".join(ARTI4D_METHODS),
            "gt_interaction_cue": True,
            "gt_part_labels_in_inference": False,
            "notes": "Official EASY interaction; two-state methods remain unsupported because native calibrated start/end multiview scans are absent.",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = manifest_rows()
    csv_path = args.output_dir / "manifest.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    protocol = {
        "comparison_type": "bidirectional_native_and_cross_domain",
        "primary_questions": [
            "Can ArtiPoint run on an aligned PartNet object where Track2Art is reliable?",
            "Can Track2Art and applicable baselines run on an official ArtiPoint interaction?",
        ],
        "scene28_role": "real_data_failure_diagnostic_only",
        "metric_policy": {
            "segmentation": ["point_iou", "ari", "rand_index", "predicted_part_count"],
            "kinematics": ["joint_type_accuracy", "axis_angle_error_deg", "axis_line_error_bbox_if_revolute"],
            "unsupported_outputs": "not_imputed",
        },
        "rows": rows,
    }
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(csv_path)


if __name__ == "__main__":
    main()
