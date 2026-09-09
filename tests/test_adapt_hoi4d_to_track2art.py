import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.adapt_hoi4d_to_track2art import _metric_extrinsics, build_episode, sequence_key


def test_sequence_key_maps_flat_archive_name():
    assert sequence_key("cam_H1_C4_N1_S1_s1_T2") == Path("cam/H1/C4/N1/S1/s1/T2")


def test_build_episode_uses_part_mask_and_raw_depth(tmp_path):
    sequence = tmp_path / "raw/cam_H1_C4_N1_S1_s1_T2"
    annotation = tmp_path / "ann"
    (sequence / "camera/recon/split_0").mkdir(parents=True)
    info = {
        "crop_intrinsic": {"fx": 10, "fy": 11, "cx": 4, "cy": 5},
        "extrinsics": [
            [[1, 0, 0, value], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            for value in (0, 1, 2, 3)
        ],
    }
    (sequence / "camera/recon/split_0/info.json").write_text(json.dumps(info))
    mask_root = annotation / "cam/H1/C4/N1/S1/s1/T2/2Dseg/mask"
    for root in (sequence / "images", sequence / "raw_depth", mask_root):
        root.mkdir(parents=True)
    pose_root = annotation / "cam/H1/C4/N1/S1/s1/T2/objpose"
    pose_root.mkdir(parents=True)
    for frame_id in range(1, 5):
        (pose_root / f"{frame_id}.json").write_text(json.dumps({
            "frameId": frame_id,
            "isEffective": 1,
            "dataList": [{"label": "Lockerbody", "center": {"x": 0.1 * (frame_id - 1), "y": 0, "z": 1}}],
        }))
    Image.new("RGB", (8, 6)).save(sequence / "images/00000.png")
    Image.new("I;16", (8, 6), 1000).save(sequence / "raw_depth/00000.png")
    Image.new("L", (8, 6), 2).save(mask_root / "00000.png")
    episode = build_episode(sequence, annotation, frame_stride=4)
    assert episode["frames"][0]["part_mask_path"].endswith("2Dseg/mask/00000.png")
    assert episode["metadata"]["depth_convention"] == "opencv-z-depth"
    assert episode["metadata"]["depth_scale_to_m"] == 0.001
    segmentation = episode["metadata"]["part_segmentation"]
    assert segmentation["ignored_part_ids"] == [2, 5]
    assert [part["part_id"] for part in segmentation["parts"]] == [1, 3, 4]
    assert [part["role"] for part in segmentation["parts"]] == ["moving", "base", "moving"]


def test_metric_extrinsics_recovers_translation_scale(tmp_path):
    annotation = tmp_path / "annotation"
    (annotation / "objpose").mkdir(parents=True)
    extrinsics = []
    for index, translation in enumerate((0.0, 10.0, 20.0, 30.0), start=1):
        matrix = [[1, 0, 0, translation], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        extrinsics.append(matrix)
        (annotation / "objpose" / f"{index}.json").write_text(json.dumps({
            "frameId": index, "isEffective": 1,
            "dataList": [{"label": "Laptopkeyboard", "center": {"x": 1 + 0.01 * translation, "y": 2, "z": 3}}],
        }))
    metric, scale = _metric_extrinsics(np.asarray(extrinsics, dtype=float), annotation, "C3")
    assert abs(scale - 0.01) < 1e-9
    assert abs(metric[-1, 0, 3] - 0.3) < 1e-9


def test_build_episode_respects_exclusive_end_frame(tmp_path):
    sequence = tmp_path / "raw/cam_H1_C4_N1_S1_s1_T2"
    annotation = tmp_path / "ann"
    (sequence / "camera/recon/split_0").mkdir(parents=True)
    info = {
        "orig_intrinsic": {"fx": 10, "fy": 11, "cx": 4, "cy": 5},
        "extrinsics": [
            [[1, 0, 0, frame_id], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            for frame_id in range(6)
        ],
    }
    (sequence / "camera/recon/split_0/info.json").write_text(json.dumps(info))
    key = annotation / "cam/H1/C4/N1/S1/s1/T2"
    for root in (sequence / "images", sequence / "raw_depth", key / "2Dseg/mask", key / "objpose"):
        root.mkdir(parents=True)
    for frame_id in range(6):
        Image.new("RGB", (8, 6)).save(sequence / f"images/{frame_id:05d}.png")
        Image.new("I;16", (8, 6), 1000).save(sequence / f"raw_depth/{frame_id:05d}.png")
        Image.new("L", (8, 6), 3).save(key / f"2Dseg/mask/{frame_id:05d}.png")
        (key / "objpose" / f"{frame_id + 1}.json").write_text(json.dumps({
            "frameId": frame_id + 1, "isEffective": 1,
            "dataList": [{"label": "Lockerbody", "center": {"x": 1 + 0.1 * frame_id, "y": 0, "z": 1}}],
        }))
    episode = build_episode(sequence, annotation, frame_stride=2, end_frame=5)
    assert [frame["action_log"]["source_frame_index"] for frame in episode["frames"]] == [0, 2, 4]
    assert episode["metadata"]["source_frame_range"] == [0, 5]
