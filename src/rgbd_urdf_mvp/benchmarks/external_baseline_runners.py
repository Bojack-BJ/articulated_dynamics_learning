"""Command planning and execution for released two-state baselines."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class BaselineRunPlan:
    method: str
    object_id: str
    oracle_gt_part_count: int | None
    commands: list[list[str]]
    output_dir: str
    protocol: str = "two_state_100view_start_end"
    oracle_inputs: tuple[str, ...] = ()
    environment: tuple[tuple[str, str], ...] = ()
    completion_artifacts: tuple[str, ...] = ()


def plan_dta_run(
    repo: Path,
    package_dir: Path,
    output_dir: Path,
    *,
    object_id: str,
    gt_part_count: int,
    python: str = "python",
    config_dir: str = "config/release",
    no_wandb: bool = True,
    correspondence_top_k: int = 30,
) -> BaselineRunPlan:
    """Build the official DTA command without modifying its implementation."""
    repo = repo.expanduser().resolve()
    package_dir = package_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_package(package_dir, "dta")
    if gt_part_count < 1:
        raise ValueError("DTA requires a positive oracle GT part count")

    # Upstream derives paths relative to a directory named "runs".
    native_output = repo / "runs" / "external_baseline_suite_v1" / object_id
    commands: list[list[str]] = []
    correspondence_dir = package_dir / "dta" / "correspondence_loftr" / "no_filter"
    if not any(correspondence_dir.glob("*.npz")):
        commands.append(
            [
                python,
                "preproc/gen_correspondence.py",
                "--data_path",
                str(package_dir / "dta"),
                "--output_path",
                str(package_dir / "dta" / "correspondence_loftr"),
                "--top_k",
                str(correspondence_top_k),
            ]
        )
    command = [
        python,
        "main.py",
        "--data_dir",
        str(package_dir / "dta"),
        "--cfg_dir",
        config_dir,
        "--num_parts",
        str(gt_part_count),
        "--save_dir",
        str(native_output),
    ]
    if no_wandb:
        command.append("--no_wandb")
    commands.append(command)
    return BaselineRunPlan(
        method="dta",
        object_id=object_id,
        oracle_gt_part_count=gt_part_count,
        commands=commands,
        output_dir=str(output_dir),
        oracle_inputs=("gt_part_count",),
        environment=(
            ("PYTHONPATH", os.pathsep.join((str(repo), str(repo / "mycuda")))),
        ),
        completion_artifacts=(
            *(
                str(
                    native_output
                    / "results"
                    / "step_0004000"
                    / f"init_part_{part_id}_clustered.obj"
                )
                for part_id in range(gt_part_count)
            ),
            str(
                native_output
                / "results"
                / "step_0004000"
                / "init_prismatic_motion.json"
            ),
            str(
                native_output
                / "results"
                / "step_0004000"
                / "init_revolute_motion.json"
            ),
        )
        if config_dir == "config/release"
        else (),
    )


def plan_artgs_run(
    repo: Path,
    package_dir: Path,
    output_dir: Path,
    *,
    object_id: str,
    gt_part_count: int,
    python: str = "python",
    dataset: str = "external_suite",
    subset: str = "aligned",
) -> BaselineRunPlan:
    """Build the released ArtGS coarse, type-predict, and full commands."""
    repo = repo.expanduser().resolve()
    package_dir = package_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_package(package_dir, "artgs")
    if gt_part_count < 1:
        raise ValueError("ArtGS requires a positive oracle GT slot count")

    source_root = repo / "data"
    native_root = repo / "outputs" / dataset / subset / object_id
    common = [
        "--dataset",
        dataset,
        "--subset",
        subset,
        "--scene_name",
        object_id,
        "--source_path",
        str(source_root),
    ]
    commands = [
        [
            python,
            "train_coarse.py",
            *common,
            "--model_path",
            str(native_root / "coarse_gs"),
            "--resolution",
            "2",
            "--iterations",
            "10000",
            "--opacity_reg_weight",
            "0.1",
            "--random_bg_color",
        ],
        [
            python,
            "train_predict.py",
            *common,
            "--model_path",
            str(native_root / "joint_predict"),
            "--eval",
            "--resolution",
            "8",
            "--iterations",
            "5000",
            "--densify_grad_threshold",
            "0.001",
            "--coarse_name",
            "coarse_gs",
            "--random_bg_color",
        ],
        [
            python,
            "train.py",
            *common,
            "--model_path",
            str(native_root / "artgs"),
            "--eval",
            "--resolution",
            "1",
            "--iterations",
            "20000",
            "--coarse_name",
            "coarse_gs",
            "--seed",
            "0",
            "--use_art_type_prior",
            "--random_bg_color",
            "--densify_grad_threshold",
            "0.001",
        ],
    ]
    return BaselineRunPlan(
        method="artgs",
        object_id=object_id,
        oracle_gt_part_count=gt_part_count,
        commands=commands,
        output_dir=str(output_dir),
        oracle_inputs=("gt_part_count",),
        environment=(
            (
                "PYTHONPATH",
                os.pathsep.join(
                    (
                        str(repo / "submodules" / "diff-gaussian-rasterization"),
                        str(repo / "submodules" / "simple-knn"),
                        str(repo),
                    )
                ),
            ),
        ),
    )


def plan_gaussianart_run(
    repo: Path,
    package_dir: Path,
    output_dir: Path,
    *,
    object_id: str,
    gt_part_count: int,
    python: str = "python",
    gpu: int = 0,
) -> BaselineRunPlan:
    """Build GaussianArt's official depth-semantic initialization and training run."""
    repo = repo.expanduser().resolve()
    package_dir = package_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_package(package_dir, "gaussianart")
    if gt_part_count < 2:
        raise ValueError("GaussianArt requires at least two oracle part slots")

    command = [
        python,
        "run.py",
        "--model_id",
        object_id,
        "--gpu",
        str(gpu),
        "--root_dir",
        str(package_dir / "gaussianart"),
        "--save_dir",
        str(output_dir / "raw"),
    ]
    return BaselineRunPlan(
        method="gaussianart",
        object_id=object_id,
        oracle_gt_part_count=gt_part_count,
        commands=[command],
        output_dir=str(output_dir),
        oracle_inputs=("part_count", "part_semantic_initialization", "gt_motion_metadata"),
    )


def plan_paris_run(
    repo: Path,
    package_dir: Path,
    output_dir: Path,
    *,
    object_id: str,
    gt_part_count: int,
    python: str = "python",
    config: str = "configs/se3.yaml",
) -> BaselineRunPlan:
    """Build the non-type-oracle PARIS SE(3) fitting command."""
    repo = repo.expanduser().resolve()
    package_dir = package_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_package(package_dir, "paris")
    if gt_part_count != 2:
        raise ValueError("PARIS supports only one static and one moving part")

    source = f"external_suite/{object_id}"
    resolution_overrides: list[str] = []
    image_paths = sorted((package_dir / "paris" / "start" / "train").glob("*.png"))
    if image_paths:
        with Image.open(image_paths[0]) as image:
            width, height = image.size
        resolution_overrides.append(f"dataset.img_wh=[{width},{height}]")
    command = [
        python,
        "launch.py",
        "--train",
        "--config",
        config,
        f"source={source}",
        f"dataset.root_dir={package_dir / 'paris'}",
        (
            "model.motion_gt_path="
            f"{package_dir / 'paris' / 'textured_objs' / 'trans.json'}"
        ),
        f"exp_dir={output_dir / 'raw'}",
        *resolution_overrides,
        "model.ray_chunk=1024",
    ]
    return BaselineRunPlan(
        method="paris",
        object_id=object_id,
        oracle_gt_part_count=None,
        commands=[command],
        output_dir=str(output_dir),
        oracle_inputs=(),
        environment=(
            # PARIS/nerfacc JIT-compiles CUDA extensions. A per-run cache
            # prevents concurrent baseline jobs from racing on one lock file.
            ("TORCH_EXTENSIONS_DIR", str(output_dir / "torch_extensions")),
        ),
    )


def plan_ditto_run(
    repo: Path,
    package_dir: Path,
    output_dir: Path,
    *,
    object_id: str,
    gt_part_count: int,
    checkpoint: Path,
    python: str = "python",
) -> BaselineRunPlan:
    """Build the repository-side Ditto inference adapter command."""
    repo = repo.expanduser().resolve()
    package_dir = package_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    _validate_package(package_dir, "ditto")
    if gt_part_count != 2:
        raise ValueError("Ditto supports only one static and one moving part")
    adapter = Path(__file__).resolve().parents[3] / "scripts" / "run_ditto_adapter.py"
    if not adapter.is_file():
        raise FileNotFoundError(f"Ditto adapter not found: {adapter}")
    command = [
        python,
        str(adapter),
        "--repo",
        str(repo),
        "--checkpoint",
        str(checkpoint.expanduser().resolve()),
        "--input",
        str(package_dir / "ditto" / "two_state_points.npz"),
        "--output-dir",
        str(output_dir / "raw"),
    ]
    return BaselineRunPlan(
        method="ditto",
        object_id=object_id,
        oracle_gt_part_count=None,
        commands=[command],
        output_dir=str(output_dir),
        oracle_inputs=(),
    )


def plan_videoartgs_run(
    repo: Path,
    source_root: Path,
    output_dir: Path,
    *,
    object_id: str,
    joint_inventory_source: str,
    python: str = "python",
    dataset: str = "external_suite",
    subset: str = "aligned",
    start_stage: int = 0,
    include_reconstruction_export: bool = True,
) -> BaselineRunPlan:
    """Build the three official VideoArtGS optimization stages."""
    repo = repo.expanduser().resolve()
    source_root = source_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    scene = source_root / dataset / subset / object_id
    required = ("data.npz", "filtered.npz", "joint_infos.json")
    missing = [scene / name for name in required if not (scene / name).exists()]
    if missing:
        raise FileNotFoundError(
            "VideoArtGS package is incomplete: " + ", ".join(map(str, missing))
        )
    if joint_inventory_source not in {"vlm", "manual", "gt_oracle"}:
        raise ValueError("joint_inventory_source must be vlm, manual, or gt_oracle")

    if start_stage not in {0, 1, 2}:
        raise ValueError("start_stage must be 0, 1, or 2")
    # Official VideoArtGS resolves --coarse_name/--deform_name relative to the
    # scene directory, so intermediate checkpoints must live beside the input.
    init_path = scene / "init"
    final_path = scene / "final"
    common = ["--dataset", dataset, "--subset", subset, "--scene_name", object_id]
    stage_commands = [
        [
            python, "init_cano.py", *common,
            # VideoArtGS appends dataset/subset/scene_name inside each entrypoint.
            "--source_path", str(source_root),
            "--model_path", str(init_path),
            "--resolution", "1", "--iterations", "20000",
            "--metric_depth_loss_weight", "1.0",
            "--densify_grad_threshold", "0.0004", "--random_bg_color",
        ],
        [
            python, "init_deform.py", *common,
            "--source_path", str(source_root),
            "--model_path", str(init_path),
            "--iterations", "10000", "--seed", "0",
        ],
        [
            python, "train.py", *common,
            "--source_path", str(source_root),
            "--model_path", str(final_path),
            "--resolution", "2", "--iterations", "20000",
            "--densify_grad_threshold", "0.0004",
            "--coarse_name", "init", "--deform_name", "init",
            "--seed", "0", "--metric_depth_loss_weight", "1.0",
            "--random_bg_color", "--track_loss_weight", "0.5",
        ],
    ][start_stage:]
    if include_reconstruction_export:
        stage_commands.append(
            [
                python,
                "render.py",
                *common,
                "--source_path",
                # Unlike the training entrypoints, render.py does not append
                # dataset/subset/scene_name before loading scene metadata.
                str(scene),
                "--model_path",
                str(final_path),
                "--iteration",
                # The released deform loader does not resolve iteration_best
                # when -1 is passed, even though the Gaussian loader does.
                "20000",
                "--mode",
                "recon",
            ]
        )
    oracle_inputs = (
        ("joint_count", "joint_types", "parent_topology")
        if joint_inventory_source == "gt_oracle"
        else ()
    )
    return BaselineRunPlan(
        method="videoartgs",
        object_id=object_id,
        oracle_gt_part_count=None,
        commands=stage_commands,
        output_dir=str(output_dir),
        protocol="continuous_monocular_video",
        oracle_inputs=oracle_inputs,
        environment=(
            ("PYTHONUTF8", "1"),
            ("TMPDIR", str(output_dir / "_build" / "tmp")),
            (
                "TORCH_EXTENSIONS_DIR",
                str(output_dir / "_build" / "torch_extensions"),
            ),
            ("MAX_JOBS", "4"),
        ),
    )


def write_run_plan(plan: BaselineRunPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(plan), indent=2) + "\n", encoding="utf-8")


def execute_run_plan(
    plan: BaselineRunPlan,
    *,
    repo: Path,
    metrics_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    """Execute a plan and persist transparent status/failure metadata."""
    started = time.monotonic()
    payload: dict[str, Any] = {
        "schema": "external-baseline-result-v1",
        "method": plan.method,
        "object_id": plan.object_id,
        "protocol": plan.protocol,
        "status": "planned" if dry_run else "running",
        "oracle": {
            "gt_part_count": plan.oracle_gt_part_count,
            "used_for_inference": bool(
                plan.oracle_gt_part_count is not None or plan.oracle_inputs
            ),
            "inputs": list(plan.oracle_inputs),
        },
        "commands": plan.commands,
        "supported_metrics": {},
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        payload["runtime_s"] = 0.0
        metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload

    try:
        environment = os.environ.copy()
        environment.update(dict(plan.environment))
        for name in ("TMPDIR", "TORCH_EXTENSIONS_DIR"):
            if value := environment.get(name):
                Path(value).mkdir(parents=True, exist_ok=True)
        for command in plan.commands:
            subprocess.run(command, cwd=repo, env=environment, check=True)
        payload["status"] = "completed_unadapted"
        payload["note"] = (
            "Official inference completed; output metric adaptation remains required."
        )
    except Exception as exc:
        completed_artifacts = bool(plan.completion_artifacts) and all(
            Path(path).exists() for path in plan.completion_artifacts
        )
        if completed_artifacts:
            payload["status"] = "completed_with_exporter_failure"
            payload["warning_stage"] = "official_process_teardown"
            payload["warning"] = f"{type(exc).__name__}: {exc}"
            payload["note"] = (
                "All declared native prediction artifacts exist; the official "
                "process failed only after producing the evaluation inputs."
            )
        else:
            payload["status"] = "failed"
            payload["failure_stage"] = "official_inference"
            payload["exception"] = f"{type(exc).__name__}: {exc}"
    payload["runtime_s"] = time.monotonic() - started
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _validate_package(package_dir: Path, method: str) -> None:
    manifest = package_dir / "package_manifest.json"
    oracle = package_dir / "oracle_requirements.json"
    method_dir = package_dir / method
    missing = [path for path in (manifest, oracle, method_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Two-state package is incomplete: " + ", ".join(map(str, missing))
        )
