#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


OPTIMIZED_STAGE_COLUMNS = [
    "record-mujoco",
    "fuse-pointcloud",
    "track-part-pixels",
    "estimate-part-poses",
    "infer-joints",
    "export-inferred-articulation",
    "visualize-pointcloud",
    "identify-dynamics",
    "identify-dynamics-mjx",
]

FEEDFORWARD_CLIENT_COLUMNS = [
    "prepare_generation_images_s",
    "build_payload_s",
    "send_s",
    "wait_until_completed_s",
    "download_result_s",
    "download_result_base64_s",
    "zip_write_s",
    "zip_base64_decode_write_s",
    "unpack_zip_s",
    "total_s",
]

FEEDFORWARD_SERVER_COLUMNS = [
    "hunyuan3d_s",
    "particulate_s",
    "particulate_manifest_total_s",
    "packaging_s",
    "copy_to_output_root_s",
    "zip_create_s",
    "zip_base64_encode_s",
    "total_s",
]

PARTICULATE_COLUMNS = [
    "mesh_prepare_s",
    "subprocess_s",
    "collect_artifacts_s",
    "total_s",
]

FEEDFORWARD_RESOURCE_STAGES = ["hunyuan3d", "particulate"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize optimized/feedforward articulation timing per object.",
    )
    parser.add_argument("--batch-config", type=Path, default=None, help="Optional TSV with object_id in column 3.")
    parser.add_argument("--optimized-root", type=Path, default=None, help="Batch optimized output root.")
    parser.add_argument("--optimized-log-dir", type=Path, default=None, help="Batch log dir; defaults to <optimized-root>/_batch_logs.")
    parser.add_argument("--feedforward-root", type=Path, default=None, help="Feedforward output root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <feedforward-root>/_evaluation or <optimized-root>/_evaluation.",
    )
    args = parser.parse_args()

    optimized_root = _resolve_optional(args.optimized_root)
    feedforward_root = _resolve_optional(args.feedforward_root)
    if optimized_root is None and feedforward_root is None:
        raise SystemExit("Provide at least one of --optimized-root or --feedforward-root.")

    output_dir = _resolve_output_dir(args.output_dir, optimized_root, feedforward_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    object_ids = _object_ids(args.batch_config, optimized_root, feedforward_root)
    optimized_batch_timing = _load_optimized_batch_timing(optimized_root, args.optimized_log_dir)
    optimized_rows = _load_optimized_rows(object_ids, optimized_root, args.optimized_log_dir)
    feedforward_rows = _load_feedforward_rows(object_ids, feedforward_root)
    rows = [_merge_rows(object_id, optimized_rows.get(object_id), feedforward_rows.get(object_id)) for object_id in object_ids]

    payload = {
        "source": "articulation-timing-summary",
        "batch_config": str(args.batch_config.resolve()) if args.batch_config else None,
        "optimized_root": str(optimized_root) if optimized_root else None,
        "feedforward_root": str(feedforward_root) if feedforward_root else None,
        "optimized_batch_timing": optimized_batch_timing,
        "summary": _summary(rows),
        "notes": [
            "Optimized timings are read from batch log JSON lines with event=stage_timing.",
            "Optimized batch wall-clock time is read from _batch_timing.json when present. Under object/GPU parallelism, wall-clock time measures throughput and per-object stage timings measure single-object cost; do not sum per-object stage durations and compare that sum to wall-clock.",
            "Optimized totals exclude segmentation model time; they include recording/rendering, pointcloud fusion, CoTracker tracking, pose fitting, joint inference, export, and viewer serialization when those stage timings are present.",
            "CoTracker time is the track-part-pixels stage.",
            "Feedforward timings are read from remote_articulation_timing.json and particulate/particulate_result.json.",
            "Feedforward GPU memory is read from server_resource_usage sampled with nvidia-smi. Peak total VRAM is the largest memory.used sample on any GPU during that stage; peak delta VRAM subtracts that GPU's stage-start baseline and is less affected by pre-existing allocations.",
            "Optimized CoTracker GPU memory is read from part_tracks.json resource_usage.torch. On CUDA it records allocated/reserved/max allocated memory; on MPS it records current/driver/recommended memory; on CPU the backend is recorded as cpu and GPU memory fields are blank.",
            "Blank TSV fields mean that stage was skipped by resume, the artifact predates timing instrumentation, or the object has not completed.",
        ],
        "objects": rows,
    }
    json_path = output_dir / "timing_summary.json"
    tsv_path = output_dir / "timing_summary.tsv"
    md_path = output_dir / "timing_summary.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tsv(tsv_path, rows)
    _write_markdown(md_path, payload)
    print(json.dumps({"timing_summary": str(json_path.resolve())}, indent=2))
    return 0


def _resolve_optional(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve()


def _resolve_output_dir(output_dir: Path | None, optimized_root: Path | None, feedforward_root: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    if feedforward_root is not None:
        return feedforward_root / "_evaluation"
    assert optimized_root is not None
    return optimized_root / "_evaluation"


def _object_ids(batch_config: Path | None, optimized_root: Path | None, feedforward_root: Path | None) -> list[str]:
    ids: list[str] = []
    if batch_config is not None:
        for line in batch_config.expanduser().read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) >= 3 and columns[2].strip():
                ids.append(columns[2].strip())
    for root in [optimized_root, feedforward_root]:
        if root is None or not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith("_") and child.name not in ids:
                ids.append(child.name)
    return ids


def _load_optimized_rows(
    object_ids: list[str],
    optimized_root: Path | None,
    optimized_log_dir: Path | None,
) -> dict[str, dict[str, Any]]:
    if optimized_root is None:
        return {}
    log_dir = optimized_log_dir.expanduser().resolve() if optimized_log_dir is not None else optimized_root / "_batch_logs"
    rows = {}
    for object_id in object_ids:
        stage_timings: dict[str, list[float]] = defaultdict(list)
        stage_status: dict[str, str] = {}
        log_path = log_dir / f"{object_id}.log"
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") != "stage_timing":
                    continue
                stage = str(payload.get("stage", ""))
                elapsed = _float_or_none(payload.get("elapsed_s"))
                if stage and elapsed is not None:
                    stage_timings[stage].append(elapsed)
                    stage_status[stage] = str(payload.get("status", ""))
        row: dict[str, Any] = {
            "optimized_log": str(log_path) if log_path.exists() else None,
            "optimized_stage_count": sum(len(values) for values in stage_timings.values()),
        }
        tracks_path = optimized_root / object_id / "pointcloud_4d_partseg" / "part_tracks.json"
        tracks_payload = _read_json(tracks_path)
        torch_usage = (
            tracks_payload.get("resource_usage", {}).get("torch", {})
            if isinstance(tracks_payload.get("resource_usage"), dict)
            else {}
        )
        if isinstance(torch_usage, dict):
            row["optimized_track_torch_device"] = torch_usage.get("device")
            row["optimized_track_torch_backend"] = torch_usage.get("backend")
            row["optimized_track_torch_sample_count"] = torch_usage.get("sample_count")
            row["optimized_track_peak_current_allocated_mib"] = _float_or_none(
                torch_usage.get("peak_current_allocated_mib")
            )
            row["optimized_track_peak_driver_or_reserved_mib"] = _float_or_none(
                torch_usage.get("peak_driver_or_reserved_mib")
            )
            row["optimized_track_peak_max_memory_allocated_mib"] = _float_or_none(
                torch_usage.get("peak_max_memory_allocated_mib")
            )
            recommended_max_bytes = _float_or_none(torch_usage.get("recommended_max_bytes"))
            row["optimized_track_recommended_max_mib"] = (
                recommended_max_bytes / (1024.0 * 1024.0) if recommended_max_bytes is not None else None
            )
        optimized_total = 0.0
        optimized_total_has_value = False
        for stage in OPTIMIZED_STAGE_COLUMNS:
            value = _sum_or_none(stage_timings.get(stage, []))
            row[f"optimized_{_key(stage)}_s"] = value
            row[f"optimized_{_key(stage)}_status"] = stage_status.get(stage)
            if value is not None:
                optimized_total += value
                optimized_total_has_value = True
        row["optimized_total_measured_s"] = optimized_total if optimized_total_has_value else None
        row["optimized_cotracker_s"] = row.get("optimized_track_part_pixels_s")
        rows[object_id] = row
    return rows


def _load_optimized_batch_timing(optimized_root: Path | None, optimized_log_dir: Path | None) -> dict[str, Any]:
    if optimized_root is None:
        return {}
    log_dir = optimized_log_dir.expanduser().resolve() if optimized_log_dir is not None else optimized_root / "_batch_logs"
    timing_path = log_dir / "_batch_timing.json"
    payload = _read_json(timing_path)
    if not payload:
        return {"path": str(timing_path), "available": False}
    return {"path": str(timing_path), "available": True, **payload}


def _load_feedforward_rows(object_ids: list[str], feedforward_root: Path | None) -> dict[str, dict[str, Any]]:
    if feedforward_root is None:
        return {}
    rows = {}
    for object_id in object_ids:
        object_dir = feedforward_root / object_id
        timing_path = object_dir / "remote_articulation_timing.json"
        particulate_path = object_dir / "particulate" / "particulate_result.json"
        timing = _read_json(timing_path)
        particulate = _read_json(particulate_path)
        client = timing.get("client_timings") if isinstance(timing.get("client_timings"), dict) else {}
        server = timing.get("server_timings") if isinstance(timing.get("server_timings"), dict) else {}
        resources = timing.get("server_resource_usage") if isinstance(timing.get("server_resource_usage"), dict) else {}
        particulate_timings = (
            particulate.get("timings")
            if isinstance(particulate.get("timings"), dict)
            else {}
        )
        wrapper = (
            particulate_timings.get("wrapper_reported")
            if isinstance(particulate_timings.get("wrapper_reported"), dict)
            else {}
        )
        row: dict[str, Any] = {
            "feedforward_timing": str(timing_path) if timing_path.exists() else None,
            "feedforward_particulate_result": str(particulate_path) if particulate_path.exists() else None,
            "feedforward_status": timing.get("server_status"),
            "feedforward_stage": timing.get("server_stage"),
        }
        for key in FEEDFORWARD_CLIENT_COLUMNS:
            row[f"feedforward_client_{_strip_s(key)}_s"] = _float_or_none(client.get(key))
        for key in FEEDFORWARD_SERVER_COLUMNS:
            row[f"feedforward_server_{_strip_s(key)}_s"] = _float_or_none(server.get(key))
        for stage in FEEDFORWARD_RESOURCE_STAGES:
            stage_resources = resources.get(stage) if isinstance(resources.get(stage), dict) else {}
            prefix = f"feedforward_{stage}_"
            row[f"{prefix}gpu_available"] = bool(stage_resources.get("available", False))
            row[f"{prefix}gpu_sample_count"] = stage_resources.get("sample_count")
            row[f"{prefix}peak_gpu_index"] = stage_resources.get("peak_gpu_index")
            row[f"{prefix}max_delta_gpu_index"] = stage_resources.get("max_delta_gpu_index")
            row[f"{prefix}peak_memory_used_mib"] = _float_or_none(stage_resources.get("peak_memory_used_mib"))
            row[f"{prefix}peak_memory_delta_mib"] = _float_or_none(stage_resources.get("peak_memory_delta_mib"))
        for key in PARTICULATE_COLUMNS:
            row[f"particulate_manifest_{_strip_s(key)}_s"] = _float_or_none(particulate_timings.get(key))
        row["particulate_wrapper_prepare_inputs_total_s"] = _float_or_none(wrapper.get("prepare_inputs_total_s"))
        row["particulate_wrapper_model_infer_s"] = _float_or_none(wrapper.get("model_infer_s"))
        row["particulate_wrapper_predict_mesh_total_s"] = _float_or_none(wrapper.get("predict_mesh_total_s"))
        rows[object_id] = row
    return rows


def _merge_rows(object_id: str, optimized: dict[str, Any] | None, feedforward: dict[str, Any] | None) -> dict[str, Any]:
    row: dict[str, Any] = {"object_id": object_id}
    if optimized:
        row.update(optimized)
    if feedforward:
        row.update(feedforward)
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"object_count": len(rows)}
    keys = [
        "optimized_total_measured_s",
        "optimized_cotracker_s",
        "optimized_track_peak_current_allocated_mib",
        "optimized_track_peak_driver_or_reserved_mib",
        "optimized_track_peak_max_memory_allocated_mib",
        "feedforward_client_prepare_generation_images_s",
        "feedforward_server_hunyuan3d_s",
        "feedforward_server_particulate_s",
        "feedforward_server_packaging_s",
        "feedforward_hunyuan3d_peak_memory_used_mib",
        "feedforward_hunyuan3d_peak_memory_delta_mib",
        "feedforward_particulate_peak_memory_used_mib",
        "feedforward_particulate_peak_memory_delta_mib",
        "feedforward_client_download_result_s",
        "feedforward_client_unpack_zip_s",
        "feedforward_client_total_s",
        "feedforward_server_total_s",
        "particulate_manifest_mesh_prepare_s",
        "particulate_manifest_subprocess_s",
        "particulate_manifest_total_s",
    ]
    for key in keys:
        values = [_float_or_none(row.get(key)) for row in rows]
        present = [value for value in values if value is not None]
        summary[key] = {
            "count": len(present),
            "mean_s": _mean(present),
            "median_s": _median(present),
            "min_s": min(present) if present else None,
            "max_s": max(present) if present else None,
        }
    return summary


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ["object_id"]
    for stage in OPTIMIZED_STAGE_COLUMNS:
        columns.append(f"optimized_{_key(stage)}_s")
    columns.extend(
        [
            "optimized_total_measured_s",
            "optimized_cotracker_s",
            "optimized_track_torch_device",
            "optimized_track_torch_backend",
            "optimized_track_torch_sample_count",
            "optimized_track_peak_current_allocated_mib",
            "optimized_track_peak_driver_or_reserved_mib",
            "optimized_track_peak_max_memory_allocated_mib",
            "optimized_track_recommended_max_mib",
            "feedforward_client_prepare_generation_images_s",
            "feedforward_server_hunyuan3d_s",
            "feedforward_server_particulate_s",
            "feedforward_server_packaging_s",
            "feedforward_hunyuan3d_peak_gpu_index",
            "feedforward_hunyuan3d_max_delta_gpu_index",
            "feedforward_hunyuan3d_peak_memory_used_mib",
            "feedforward_hunyuan3d_peak_memory_delta_mib",
            "feedforward_particulate_peak_gpu_index",
            "feedforward_particulate_max_delta_gpu_index",
            "feedforward_particulate_peak_memory_used_mib",
            "feedforward_particulate_peak_memory_delta_mib",
            "feedforward_client_download_result_s",
            "feedforward_client_download_result_base64_s",
            "feedforward_client_zip_write_s",
            "feedforward_client_zip_base64_decode_write_s",
            "feedforward_client_unpack_zip_s",
            "feedforward_client_total_s",
            "feedforward_server_total_s",
            "particulate_manifest_mesh_prepare_s",
            "particulate_manifest_subprocess_s",
            "particulate_manifest_collect_artifacts_s",
            "particulate_manifest_total_s",
            "particulate_wrapper_prepare_inputs_total_s",
            "particulate_wrapper_model_infer_s",
            "particulate_wrapper_predict_mesh_total_s",
            "optimized_log",
            "feedforward_timing",
            "feedforward_particulate_result",
        ]
    )
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(_format_tsv(row.get(column)) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    optimized_batch_timing = payload.get("optimized_batch_timing", {})
    rows = [
        ("Optimized total measured", "optimized_total_measured_s"),
        ("Optimized CoTracker", "optimized_cotracker_s"),
        ("Feedforward image prep", "feedforward_client_prepare_generation_images_s"),
        ("Feedforward Hunyuan3D", "feedforward_server_hunyuan3d_s"),
        ("Feedforward PARTICULATE", "feedforward_server_particulate_s"),
        ("Feedforward packaging", "feedforward_server_packaging_s"),
        ("Feedforward download", "feedforward_client_download_result_s"),
        ("Feedforward client total", "feedforward_client_total_s"),
        ("Feedforward server total", "feedforward_server_total_s"),
    ]
    memory_rows = [
        ("Optimized CoTracker peak current allocated", "optimized_track_peak_current_allocated_mib"),
        ("Optimized CoTracker peak driver/reserved", "optimized_track_peak_driver_or_reserved_mib"),
        ("Optimized CoTracker peak CUDA max allocated", "optimized_track_peak_max_memory_allocated_mib"),
        ("Hunyuan3D peak total VRAM", "feedforward_hunyuan3d_peak_memory_used_mib"),
        ("Hunyuan3D peak delta VRAM", "feedforward_hunyuan3d_peak_memory_delta_mib"),
        ("PARTICULATE peak total VRAM", "feedforward_particulate_peak_memory_used_mib"),
        ("PARTICULATE peak delta VRAM", "feedforward_particulate_peak_memory_delta_mib"),
    ]
    lines = [
        "# Articulation Timing Summary",
        "",
        f"- Optimized root: `{payload.get('optimized_root')}`",
        f"- Feedforward root: `{payload.get('feedforward_root')}`",
        f"- Objects: `{summary.get('object_count')}`",
    ]
    if isinstance(optimized_batch_timing, dict) and optimized_batch_timing.get("available"):
        lines.extend(
            [
                f"- Optimized batch wall-clock: `{_format_seconds(optimized_batch_timing.get('wall_clock_s'))} s`",
                f"- Optimized concurrency: `jobs={optimized_batch_timing.get('jobs')}`, "
                f"`tracking_jobs={optimized_batch_timing.get('tracking_jobs')}`, "
                f"`dynamics_jobs={optimized_batch_timing.get('dynamics_jobs')}`",
            ]
        )
    lines.extend(
        [
            "",
            "| Stage | Count | Mean s | Median s | Min s | Max s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in rows:
        item = summary.get(key, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(item.get("count", 0)),
                    _format_seconds(item.get("mean_s")),
                    _format_seconds(item.get("median_s")),
                    _format_seconds(item.get("min_s")),
                    _format_seconds(item.get("max_s")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## GPU Memory",
            "",
            "| Metric | Count | Mean MiB | Median MiB | Min MiB | Max MiB |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in memory_rows:
        item = summary.get(key, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(item.get("count", 0)),
                    _format_number(item.get("mean_s")),
                    _format_number(item.get("median_s")),
                    _format_number(item.get("min_s")),
                    _format_number(item.get("max_s")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Notes", ""])
    for note in payload.get("notes", []):
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _sum_or_none(values: list[float]) -> float | None:
    return sum(values) if values else None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return 0.5 * (sorted_values[mid - 1] + sorted_values[mid])


def _key(stage: str) -> str:
    return stage.replace("-", "_")


def _strip_s(key: str) -> str:
    return key[:-2] if key.endswith("_s") else key


def _format_tsv(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _format_seconds(value: Any) -> str:
    number = _float_or_none(value)
    return "" if number is None else f"{number:.3f}"


def _format_number(value: Any) -> str:
    number = _float_or_none(value)
    return "" if number is None else f"{number:.1f}"


if __name__ == "__main__":
    raise SystemExit(main())
