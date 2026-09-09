#!/usr/bin/env python3
"""Generate and evaluate the four PartNet-Mobility objects overlapping AiM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from rgbd_urdf_mvp.benchmarks.aim_overlap import (
    AIM_OBJECTS,
    AIM_REFERENCE,
    METRIC_DISCLAIMER,
    axis_distribution,
    benchmark_metadata,
    joint_success,
    select_explicit_objects,
    sequence_plan,
    validate_object_metadata,
)
from rgbd_urdf_mvp.kinematics.joint_inference import _joint_defs_from_mjcf
from rgbd_urdf_mvp.kinematics.analytic_joint_axis import AnalyticAxisConfig, estimate_analytic_joint_axis
from rgbd_urdf_mvp.kinematics.pairwise_relation_head import extract_gt_relations
from rgbd_urdf_mvp.perception.cotracker_features import load_cotracker_feature_map
from rgbd_urdf_mvp.perception.motion_part_slots import _sample_from_artifact
from rgbd_urdf_mvp.perception.motion_part_slots import evaluate_slot_assignments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("generate", "evaluate", "report", "all"), default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("cuda", "cpu", "mps", "auto"), default="cuda")
    parser.add_argument("--cotracker-repo", type=Path, required=True)
    parser.add_argument("--cotracker-checkpoint", type=Path, required=True)
    parser.add_argument("--slot-model", type=Path, required=True)
    parser.add_argument("--relation-full-slot", type=Path, required=True)
    parser.add_argument("--relation-decoder-only", type=Path)
    parser.add_argument("--relation-frozen", type=Path)
    parser.add_argument("--duration-s", type=float, default=8.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--tracking-frame-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--simultaneous-stress-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-viewers", action="store_true")
    parser.add_argument("--min-rotation-deg", type=float, default=8.0)
    parser.add_argument("--min-translation-m", type=float, default=0.02)
    return parser.parse_args()


def load_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    rows = payload.get("objects", payload) if isinstance(payload, dict) else payload
    return select_explicit_objects(rows)


def inspect_model(row: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(row["model_path"]).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    root = ET.parse(model_path).getroot()
    joints = _joint_defs_from_mjcf(model_path)
    movable = []
    parent_by_body: dict[str, str | None] = {}
    for body in root.findall(".//body"):
        name = str(body.get("name", ""))
        parent = next(
            (candidate for candidate in root.findall(".//body") if body in list(candidate)), None
        )
        parent_by_body[name] = parent.get("name") if parent is not None else None
        for element in body.findall("joint"):
            name_joint = str(element.get("name", ""))
            joint = joints.get(name_joint)
            kind = str(element.get("type", "hinge"))
            kind = "revolute" if kind == "hinge" else "prismatic" if kind == "slide" else kind
            if kind not in {"revolute", "prismatic"}:
                continue
            range_values = [float(value) for value in element.get("range", "0 0").split()]
            movable.append({
                "name": name_joint,
                "joint_type": kind,
                "parent_body": parent_by_body.get(str(body.get("name", ""))),
                "child_body": str(body.get("name", "")),
                "axis": (joint or {}).get("axis_world"),
                "pivot": (joint or {}).get("pivot_world"),
                "range": range_values,
            })
    return {
        "object_id": str(row["object_id"]),
        "source_category": str(row.get("source_category", row.get("category", "unknown"))),
        "model_path": str(model_path),
        "total_part_count": 1 + len(movable),
        "movable_joint_count": len(movable),
        "movable_joints": movable,
        "parent_child_graph": [
            {"parent": joint["parent_body"], "child": joint["child_body"], "joint": joint["name"]}
            for joint in movable
        ],
    }


def run(command: list[str], *, env: dict[str, str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, check=True, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return time.perf_counter() - started


def record_command(args: argparse.Namespace, row: dict[str, Any], plan: dict[str, Any], output_dir: Path) -> list[str]:
    command = [
        args.python, "-m", "rgbd_urdf_mvp", "record-mujoco", str(row["model_path"]),
        "--category", str(row.get("category", "partnet")), "--object-id", str(row["object_id"]),
        "--output-dir", str(output_dir), "--duration-s", str(args.duration_s), "--fps", str(args.fps),
        "--width", "480", "--height", "352", "--disable-gravity", "--disable-target-collision",
        "--perturbation-scale", "0.0", "--rgb-format", "png", "--depth-format", "png",
        "--mask-format", "png", "--camera-distance", "3.0", "--auto-camera-fit",
        "--camera-fit-fill-ratio", "0.42", "--camera-mode", "triview",
        "--camera-triview-spacing-deg", "60", "--camera-elevation-deg", "-15",
        "--camera-fovy-deg", "60", "--lookat", "0", "0", "0", "--video",
        "--segmentation-masks", "--part-segmentation-masks", "--seed", str(args.seed),
    ]
    if plan["mode"] == "staggered":
        command.extend(["--control-mode", "staggered", "--staggered-max-acceleration", "50.0", "--all-joints"])
    elif plan["mode"] == "simultaneous":
        command.extend(["--control-mode", "free", "--all-joints", "--auto-initial-qvel-from-limits"])
    else:
        command.extend([
            "--control-mode", "track", "--joint-name", str(plan["joint_names"][0]),
        ])
    return command


def joint_motion_from_episode(path: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    episode = json.loads(path.read_text(encoding="utf-8"))
    frames = episode.get("frames", [])
    output = []
    for joint in metadata["movable_joints"]:
        name = joint["name"]
        values = []
        for frame in frames:
            state = frame.get("action_log", {}).get("joint_positions", {})
            if isinstance(state, dict) and name in state:
                values.append(float(state[name]))
        excursion = max(values) - min(values) if values else 0.0
        threshold = math.radians(8.0) if joint["joint_type"] == "revolute" else 0.02
        output.append({
            "joint_name": name,
            "joint_type": joint["joint_type"],
            "frame_count": len(values),
            "motion_excursion": excursion,
            "motion_units": "rad" if joint["joint_type"] == "revolute" else "m",
            "passes_minimum_excitation": excursion >= threshold,
        })
    return output


def generate(args: argparse.Namespace, rows: list[dict[str, Any]], env: dict[str, str]) -> None:
    generation = {
        "benchmark": benchmark_metadata(),
        "config": {
            "duration_s": args.duration_s, "fps": args.fps,
            "tracking_frame_stride": args.tracking_frame_stride, "seed": args.seed,
            "camera": {"mode": "triview", "spacing_deg": 60, "elevation_deg": -15, "resolution": [480, 352]},
        },
        "objects": [],
    }
    for spec, row in zip(AIM_OBJECTS, rows):
        metadata = inspect_model(row)
        validate_object_metadata(spec, metadata)
        object_root = args.output_root / spec.object_id
        object_root.mkdir(parents=True, exist_ok=True)
        (object_root / "articulation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        plans = sequence_plan(metadata["movable_joints"], args.simultaneous_stress_test)
        object_summary = {"object_id": spec.object_id, "metadata": metadata, "sequences": []}
        for plan in plans:
            sequence_root = object_root / "sequences" / plan["name"]
            episode = sequence_root / spec.object_id / "episode.json"
            if not (args.resume and episode.is_file()):
                run(record_command(args, row, plan, sequence_root), env=env, log=args.output_root / "logs" / f"{spec.object_id}.{plan['name']}.record.log")
            motion = joint_motion_from_episode(episode, metadata)
            if plan["mode"] == "isolated":
                target = plan["joint_names"][0]
                audit = next(item for item in motion if item["joint_name"] == target)
                if not audit["passes_minimum_excitation"]:
                    raise RuntimeError(f"Insufficient isolated excitation: {spec.object_id}/{target}: {audit}")
                isolated_artifact = sequence_root / spec.object_id / "pointcloud_4d_partseg"
                isolated_tracks = isolated_artifact / "part_tracks.json"
                isolated_features = isolated_artifact / "cotracker_features.npz"
                if not (args.resume and isolated_tracks.is_file() and isolated_features.is_file()):
                    run(track_command(args, episode, isolated_tracks), env=env, log=args.output_root / "logs" / f"{spec.object_id}.{plan['name']}.track.log")
                observability = analytic_observability(
                    isolated_tracks, isolated_features, target,
                    min_rotation_deg=args.min_rotation_deg,
                    min_translation_m=args.min_translation_m,
                )
                if not observability["passes_observability"]:
                    raise RuntimeError(f"Unobservable isolated joint: {spec.object_id}/{target}: {observability}")
            else:
                observability = None
            object_summary["sequences"].append({
                **plan, "episode_path": str(episode), "joint_motion": motion,
                "observability": observability,
            })
        combined = object_root / "sequences" / "combined_sequential" / spec.object_id / "episode.json"
        artifact = object_root / "combined" / "pointcloud_4d_partseg"
        tracks = artifact / "part_tracks.json"
        features = artifact / "cotracker_features.npz"
        if not (args.resume and tracks.is_file() and features.is_file()):
            run(track_command(args, combined, tracks), env=env, log=args.output_root / "logs" / f"{spec.object_id}.track.log")
        fusion_manifest = artifact / "fusion_manifest.json"
        if not (args.resume and fusion_manifest.is_file()):
            run([
                args.python, "-m", "rgbd_urdf_mvp", "fuse-pointcloud", str(combined),
                "--output-dir", str(artifact), "--pixel-stride", "8", "--voxel-size-m", "0.02",
                "--no-shared-pose-fallback",
            ], env=env, log=args.output_root / "logs" / f"{spec.object_id}.fuse.log")
        object_summary["combined_tracks"] = str(tracks)
        object_summary["combined_features"] = str(features)
        object_summary["fusion_manifest"] = str(fusion_manifest)
        generation["objects"].append(object_summary)
        print(json.dumps({"generated": spec.object_id, "sequences": len(plans)}, indent=2), flush=True)
    (args.output_root / "generation_manifest.json").write_text(json.dumps(generation, indent=2) + "\n", encoding="utf-8")
    write_manifest(args.output_root / "combined_manifest.tsv", generation["objects"])


def track_command(args: argparse.Namespace, episode: Path, output_tracks: Path) -> list[str]:
    return [
        args.python, "-m", "rgbd_urdf_mvp", "track-part-pixels", str(episode),
        "--output-json", str(output_tracks), "--cotracker-repo", str(args.cotracker_repo),
        "--cotracker-checkpoint", str(args.cotracker_checkpoint), "--device", args.device,
        "--reference-frame", "-1", "--frame-stride", str(args.tracking_frame_stride),
        "--seed-stride-px", "12", "--max-tracks-per-part-view", "192",
        "--visibility-threshold", "0.5", "--export-cotracker-features",
    ]


def analytic_observability(
    tracks_path: Path, features_path: Path, joint_name: str, *,
    min_rotation_deg: float, min_translation_m: float,
) -> dict[str, Any]:
    import numpy as np

    artifact = json.loads(tracks_path.read_text(encoding="utf-8"))
    sample = _sample_from_artifact(
        artifact, load_cotracker_feature_map(features_path),
        object_id=str(artifact.get("object_instance_id", tracks_path.parent.name)),
        require_labels=True, canonicalize_geometry=False,
    )
    relations = extract_gt_relations(artifact, sample)
    relation = next((row for row in relations if row["joint_name"] == joint_name), None)
    if relation is None:
        return {"passes_observability": False, "reason": "joint_relation_not_recovered", "joint_name": joint_name}
    labels = np.asarray(sample["labels"], dtype=int)
    parent = labels == int(relation["parent_label"])
    child = labels == int(relation["child_label"])
    points = np.asarray(sample["points"], dtype=float)
    visibility = np.asarray(sample["visibility"], dtype=bool)
    center = np.asarray(sample["canonical_center_m"], dtype=float)
    scale = max(float(sample["canonical_scale_m"]), 1e-8)
    canonical = (points - center) / scale
    config = AnalyticAxisConfig(
        min_rotation_rad=math.radians(2.0),
        min_total_rotation_rad=math.radians(min_rotation_deg),
        min_total_translation=min_translation_m / scale,
    )
    estimate = estimate_analytic_joint_axis(
        canonical[parent], visibility[parent], canonical[child], visibility[child],
        str(relation["joint_type"]), config,
    ).to_dict()
    common_frames = int(np.sum(np.any(visibility[parent], axis=0) & np.any(visibility[child], axis=0)))
    return {
        "joint_name": joint_name,
        "joint_type": relation["joint_type"],
        "parent_track_count": int(parent.sum()),
        "child_track_count": int(child.sum()),
        "common_parent_child_frame_count": common_frames,
        "valid_frame_count": int(estimate.get("valid_frame_count", 0)),
        "total_motion": estimate.get("total_motion"),
        "rigid_fit_residual": estimate.get("residual"),
        "analytic_confidence": estimate.get("confidence"),
        "analytic_reason": estimate.get("reason"),
        "passes_observability": bool(estimate.get("valid")),
    }


def write_manifest(path: Path, objects: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["object_id", "tracks_path", "features_npz", "split"], delimiter="\t")
        writer.writeheader()
        for item in objects:
            writer.writerow({
                "object_id": item["object_id"], "tracks_path": item["combined_tracks"],
                "features_npz": item["combined_features"], "split": "test",
            })


def evaluate(args: argparse.Namespace, env: dict[str, str]) -> None:
    manifest = args.output_root / "combined_manifest.tsv"
    relation_models = {
        "full_slot": args.relation_full_slot,
        "decoder_only": args.relation_decoder_only,
        "frozen": args.relation_frozen,
    }
    for method, relation_model in relation_models.items():
        if relation_model is None or not relation_model.expanduser().exists():
            continue
        output = args.output_root / "evaluation" / method
        output.mkdir(parents=True, exist_ok=True)
        run([
            args.python, "scripts/evaluate_analytic_joint_axis_oracles.py", str(manifest),
            str(args.slot_model), str(relation_model), "--catalog", str(args.catalog),
            "--output-dir", str(output / "axis_oracles"), "--device", args.device,
        ], env=env, log=args.output_root / "logs" / f"evaluate.{method}.log")
        for row in csv.DictReader(manifest.open("r", encoding="utf-8"), delimiter="\t"):
            object_id = row["object_id"]
            object_output = output / object_id
            relation_json = object_output / "slot_relations.json"
            run([
                args.python, "-m", "rgbd_urdf_mvp", "infer-slot-relation-head",
                row["tracks_path"], row["features_npz"], str(args.slot_model), str(relation_model),
                "--output-json", str(relation_json), "--device", args.device,
            ], env=env, log=args.output_root / "logs" / f"{object_id}.{method}.infer.log")
            materialize_prediction_and_viewer(args, row, relation_json, object_output, env)


def materialize_prediction_and_viewer(
    args: argparse.Namespace, row: dict[str, str], relation_path: Path, output: Path, env: dict[str, str]
) -> None:
    tracks = json.loads(Path(row["tracks_path"]).read_text(encoding="utf-8"))
    relation = json.loads(relation_path.read_text(encoding="utf-8"))
    assignments = relation["track_slot_assignments"]
    for track, slot in zip(tracks.get("tracks", []), assignments):
        track.setdefault("original_part_id", int(track.get("part_id", -1)))
        track["part_id"] = int(slot)
    tracks["motion_segmentation"] = {
        "source": "slot_relation_checkpoint", "active_slots": relation["active_slots"],
        "slot_existence_probabilities": relation["slot_existence_probabilities"],
    }
    prediction = output / "motion_part_tracks_slots.json"
    prediction.parent.mkdir(parents=True, exist_ok=True)
    prediction.write_text(json.dumps(tracks, indent=2) + "\n", encoding="utf-8")
    if args.skip_viewers:
        return
    joint_inference = output / "joint_inference_relation.json"
    metadata_path = args.output_root / row["object_id"] / "articulation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    segmentation = tracks.get("original_part_segmentation", tracks.get("part_segmentation", {}))
    body_to_part = {
        str(part.get("body_name")): int(part.get("part_id", -1))
        for part in segmentation.get("parts", []) if isinstance(part, dict)
    }
    predicted_joints = [
        {
            "name": f"pred_slot_{edge['parent_slot_id']}_{edge['child_slot_id']}",
            "joint_type": edge["joint_type"],
            "parent_part_id": edge["parent_slot_id"],
            "child_part_id": edge["child_slot_id"],
            "axis": edge["axis_world"],
            "pivot": edge["axis_line_point_world"],
        }
        for edge in relation.get("selected_edges", [])
    ]
    gt_joints = [
        {
            "name": f"GT_{joint['name']}",
            "joint_type": joint["joint_type"],
            "parent_part_id": body_to_part.get(str(joint.get("parent_body")), -1),
            "child_part_id": body_to_part.get(str(joint.get("child_body")), -1),
            "axis": joint["axis"],
            "pivot": joint["pivot"],
        }
        for joint in metadata.get("movable_joints", [])
        if joint.get("axis") is not None and joint.get("pivot") is not None
    ]
    joint_inference.write_text(
        json.dumps({"joints": predicted_joints + gt_joints}, indent=2) + "\n",
        encoding="utf-8",
    )
    viewer = output / "viewer.html"
    viewer_command = [
        args.python, "-m", "rgbd_urdf_mvp", "visualize-object-mask-flow-html", str(prediction),
        "--output-html", str(viewer), "--max-tracks", "1200", "--axis-remap", "x,y,z",
        "--frame-stride", "1", "--joint-inference", str(joint_inference),
    ]
    object_root = args.output_root / row["object_id"]
    fusion_manifest = object_root / "combined" / "pointcloud_4d_partseg" / "fusion_manifest.json"
    episode = object_root / "sequences" / "combined_sequential" / row["object_id"] / "episode.json"
    if fusion_manifest.is_file():
        viewer_command.extend(["--background-fusion-manifest", str(fusion_manifest)])
    if episode.is_file():
        viewer_command.extend(["--mjcf-replay-episode", str(episode)])
    run(viewer_command, env=env, log=args.output_root / "logs" / f"{row['object_id']}.{output.parent.name}.viewer.log")


def report(args: argparse.Namespace) -> None:
    generation = json.loads((args.output_root / "generation_manifest.json").read_text(encoding="utf-8"))
    object_rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for method_root in sorted((args.output_root / "evaluation").glob("*")):
        if not method_root.is_dir():
            continue
        per_joint_path = method_root / "axis_oracles" / "analytic_axis_per_joint.json"
        if not per_joint_path.is_file():
            continue
        method_joints = json.loads(per_joint_path.read_text(encoding="utf-8"))["joints"]
        joint_rows.extend({"method": method_root.name, **row} for row in method_joints)
        for item in generation["objects"]:
            object_id = item["object_id"]
            object_spec = next(spec for spec in AIM_OBJECTS if spec.object_id == object_id)
            prediction_path = method_root / object_id / "motion_part_tracks_slots.json"
            relation_path = method_root / object_id / "slot_relations.json"
            if not prediction_path.is_file() or not relation_path.is_file():
                failures.append({"object_id": object_id, "method": method_root.name, "reason": "missing_prediction"})
                continue
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            relation = json.loads(relation_path.read_text(encoding="utf-8"))
            generated = next(row for row in generation["objects"] if row["object_id"] == object_id)
            sample = _sample_from_artifact(
                prediction,
                load_cotracker_feature_map(Path(generated["combined_features"])),
                object_id=object_id,
                require_labels=True,
                canonicalize_geometry=False,
            )
            labels = [int(value) for value in sample["labels"]]
            assignments = [int(track["part_id"]) for track in prediction["tracks"]]
            segmentation = evaluate_slot_assignments(assignments, labels)
            visible_gt_part_count = int(segmentation["gt_part_count"])
            segmentation["visible_gt_part_count"] = visible_gt_part_count
            segmentation["gt_part_count"] = object_spec.total_part_count
            segmentation["part_count_exact"] = (
                int(segmentation["predicted_part_count"]) == object_spec.total_part_count
            )
            segmentation["unobserved_gt_part_count"] = max(
                0, object_spec.total_part_count - visible_gt_part_count
            )
            match_rows = list(segmentation.get("matching", []))
            raw_part_to_label = {int(key): int(value) for key, value in sample["raw_part_to_label"].items()}
            part_segmentation = prediction.get("original_part_segmentation", prediction.get("part_segmentation", {}))
            static_gt_parts = {
                raw_part_to_label[int(part["part_id"])]
                for part in part_segmentation.get("parts", [])
                if str(part.get("part_name", "")).lower() == "base"
                and int(part.get("part_id", -1)) in raw_part_to_label
            }
            static_matches = [row for row in match_rows if int(row["gt_part"]) in static_gt_parts]
            moving_matches = [row for row in match_rows if int(row["gt_part"]) not in static_gt_parts]
            selected = [
                row for row in method_joints
                if row["object_id"] == object_id and row["setting"] == "full_neural"
            ]
            gt_slot_pairs = {
                (int(row["parent_slot"]), int(row["child_slot"]))
                for row in selected
            }
            detected_slot_pairs = {
                (int(row["parent_slot_id"]), int(row["child_slot_id"]))
                for row in relation.get("selected_edges", [])
            }
            edge_tp = len(gt_slot_pairs & detected_slot_pairs)
            detected = len(detected_slot_pairs)
            edge_precision = edge_tp / max(1, detected)
            edge_recall = edge_tp / max(1, object_spec.movable_joint_count)
            edge_f1 = 2 * edge_precision * edge_recall / max(edge_precision + edge_recall, 1e-12)
            object_rows.append({
                "object_id": object_id, "method": method_root.name,
                **{key: value for key, value in segmentation.items() if not isinstance(value, list)},
                "gt_edge_count": object_spec.movable_joint_count,
                "visible_mapped_gt_edge_pair_count": len(gt_slot_pairs),
                "detected_edge_count": detected,
                "edge_precision": edge_precision, "edge_recall": edge_recall, "edge_f1": edge_f1,
                "joint_type_accuracy": sum(bool(row["type_correct"]) for row in selected) / max(1, len(selected)),
                "labeled_track_count": len(labels),
                "unlabeled_track_count": 0,
                "matched_part_metrics": match_rows,
                "static_base_mean_iou": (
                    sum(float(row["iou"]) for row in static_matches) / len(static_matches)
                    if static_matches else None
                ),
                "moving_part_mean_iou": (
                    sum(float(row["iou"]) for row in moving_matches) / len(moving_matches)
                    if moving_matches else None
                ),
                **{f"axis_{key}": value for key, value in axis_distribution(selected).items()},
                **{f"joint_success_at_{threshold}_deg": joint_success(selected, threshold) for threshold in (5, 10, 20)},
            })
            if not segmentation["part_count_exact"]:
                failures.append({"object_id": object_id, "method": method_root.name, "reason": "exact_k_failure"})
            for row in selected:
                if not row["edge_detected"] or not row["type_correct"] or (row["axis_error_deg"] or 0) > 30:
                    failures.append({
                        "object_id": object_id, "method": method_root.name, "joint_id": row["joint_id"],
                        "reason": "edge_type_or_axis_failure", "edge_detected": row["edge_detected"],
                        "type_correct": row["type_correct"], "axis_error_deg": row["axis_error_deg"],
                    })
    write_csv(args.output_root / "aim_overlap_objects.csv", object_rows)
    write_csv(args.output_root / "aim_overlap_joints.csv", joint_rows)
    summary = {
        **benchmark_metadata(), "config": generation["config"],
        "object_metrics": object_rows, "joint_metrics": joint_rows, "failures": failures,
    }
    summary["partnet_47024_consistency"] = consistency_check_47024(object_rows, joint_rows)
    if summary["partnet_47024_consistency"]["regression"]:
        failures.append({
            "object_id": "partnet_47024", "method": "full_slot",
            "reason": "consistency_regression",
            "details": summary["partnet_47024_consistency"]["reasons"],
        })
    (args.output_root / "aim_overlap_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "failure_report.json").write_text(json.dumps({"failures": failures}, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_root / "aim_overlap_comparison.md", object_rows)


def consistency_check_47024(
    object_rows: list[dict[str, Any]], joint_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    segmentation = next(
        (row for row in object_rows if row["object_id"] == "partnet_47024" and row["method"] == "full_slot"),
        None,
    )
    neural = [
        row for row in joint_rows
        if row["object_id"] == "partnet_47024" and row["method"] == "full_slot"
        and row["setting"] == "full_neural" and row.get("axis_error_deg") is not None
    ]
    analytic_revolute = next((
        row for row in joint_rows
        if row["object_id"] == "partnet_47024" and row["method"] == "full_slot"
        and row["setting"] == "oracle_gt_parts_analytic" and row["joint_type"] == "revolute"
    ), None)
    reasons = []
    if segmentation is None:
        reasons.append("missing_full_slot_result")
    else:
        if not segmentation["part_count_exact"]:
            reasons.append("Exact K changed from true")
        if float(segmentation["one_to_one_mean_iou"]) < 0.95:
            reasons.append("visible-track one-to-one IoU dropped below 0.95")
        if float(segmentation["adjusted_rand_index"]) < 0.95:
            reasons.append("ARI dropped below 0.95")
    for row in neural:
        if float(row["axis_error_deg"]) > 12.5:
            reasons.append(f"neural {row['joint_id']} axis error exceeds 12.5 degrees")
    if analytic_revolute is None or analytic_revolute.get("axis_error_deg") is None:
        reasons.append("analytic revolute result unavailable")
    elif float(analytic_revolute["axis_error_deg"]) > 12.5:
        reasons.append("analytic revolute axis error exceeds 12.5 degrees")
    return {
        "reference": {
            "predicted_parts": 3, "gt_parts": 3, "exact_k": True,
            "one_to_one_iou": 1.0, "ari": 1.0,
            "full_neural_axis_errors_deg": [2.41, 2.31],
            "analytic_revolute_axis_error_deg": 2.89,
            "analytic_revolute_axis_line_error_bbox_normalized": 0.005,
        },
        "regression": bool(reasons), "reasons": reasons,
        "possible_causes_to_audit": [
            "random seed", "rendering configuration", "motion trajectory",
            "tracking output", "slot/relation checkpoint",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(row.get(key)) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys} for row in rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# AiM-Overlap Benchmark", "", f"> **Metric warning:** {METRIC_DISCLAIMER}", "", "## Published AiM Reference", "", "| Object | AiM mesh IoU metadata | AiM revolute axis metadata |", "|---|---:|---:|"]
    for object_id, reference in AIM_REFERENCE.items():
        mesh = reference.get("mean_mesh_part_iou", reference.get("dynamic_part_mean_mesh_iou"))
        axis = reference.get("revolute_axis_error_deg", reference.get("average_revolute_axis_error_deg", reference.get("reported_revolute_axis_errors_deg")))
        lines.append(f"| {object_id} | {mesh} | {axis} |")
    lines.extend(["", "## Current Visible-Track Results", "", "| Method | Object | K pred/GT | Exact K | Track IoU | ARI | Edge F1 | Type Acc. | Axis median | JS@10 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['object_id']} | {row['predicted_part_count']}/{row['gt_part_count']} | "
            f"{row['part_count_exact']} | {row['one_to_one_mean_iou']:.3f} | {row['adjusted_rand_index']:.3f} | "
            f"{row['edge_f1']:.3f} | {row['joint_type_accuracy']:.3f} | {row['axis_median_deg']} | "
            f"{row['joint_success_at_10_deg']:.3f} |"
        )
    lines.extend(["", "The published and current segmentation columns intentionally retain different names because they are not the same metric."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = load_catalog(args.catalog)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path("src").resolve())
    if sys.platform.startswith("linux"):
        env.setdefault("MUJOCO_GL", "egl")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
    if args.stage in {"generate", "all"}:
        generate(args, rows, env)
    if args.stage in {"evaluate", "all"}:
        evaluate(args, env)
    if args.stage in {"report", "all"}:
        report(args)
    print(json.dumps({"output_root": str(args.output_root), "stage": args.stage}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
