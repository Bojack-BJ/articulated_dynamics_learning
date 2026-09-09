#!/usr/bin/env python3
"""Build the interactive Figure-1 refrigerator viewer with semantic colors."""

from pathlib import Path

from rgbd_urdf_mvp.perception.object_mask_flow_html import (
    ObjectMaskFlowHtmlBuilder,
    ObjectMaskFlowHtmlConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    recording = ROOT / "outputs/recordings_refrigerators_staged_dense_mps/refrigerator045"
    flow = ROOT / "outputs/flow_tracking_eval/refrigerators_dense_mps_object_mask/learning_features_v1/refrigerator045"
    output = ROOT / "outputs/figure1_open_panels/viewer_open_fused_pcd.html"
    result = ObjectMaskFlowHtmlBuilder().build(
        ObjectMaskFlowHtmlConfig(
            motion_tracks=flow / "predicted_slots_balanced_v2.json",
            output_html=output,
            joint_inference=flow / "joint_inference_slots_balanced_v2.json",
            background_fusion_manifest=recording / "pointcloud_4d_partseg/fusion_manifest.json",
            background_max_points=5000,
            mjcf_replay_episode=recording / "episode.json",
            mjcf_mesh_opacity=0.28,
            max_tracks=1000,
            frame_stride=1,
            trail_length=30,
            axis_remap="x,y,z",
            color_by="gt_part",
            gt_part_colors={
                0: "#858B93",
                1: "#858B93",
                2: "#397BC5",
                3: "#F28E2B",
                4: "#397BC5",
            },
        )
    )
    print(result)


if __name__ == "__main__":
    main()
