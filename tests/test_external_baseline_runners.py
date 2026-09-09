from __future__ import annotations

import json
from pathlib import Path

from rgbd_urdf_mvp.benchmarks.external_baseline_runners import (
    BaselineRunPlan,
    execute_run_plan,
    plan_artgs_run,
    plan_ditto_run,
    plan_dta_run,
    plan_gaussianart_run,
    plan_paris_run,
    plan_videoartgs_run,
)
from scripts.run_two_state_external_baseline import register_artgs_slot_count


def _package(tmp_path: Path) -> Path:
    package = tmp_path / "package"
    package.mkdir()
    (package / "package_manifest.json").write_text("{}", encoding="utf-8")
    (package / "oracle_requirements.json").write_text("{}", encoding="utf-8")
    (package / "dta").mkdir()
    (package / "artgs").mkdir()
    (package / "gaussianart").mkdir()
    (package / "paris").mkdir()
    (package / "ditto").mkdir()
    npz = package / "ditto" / "two_state_points.npz"
    npz.write_bytes(b"npz")
    return package


def test_dta_plan_labels_oracle_and_uses_runs_path(tmp_path: Path) -> None:
    plan = plan_dta_run(
        tmp_path / "DigitalTwinArt",
        _package(tmp_path),
        tmp_path / "result",
        object_id="partnet_1",
        gt_part_count=3,
    )
    command = plan.commands[0]
    assert command[1] == "preproc/gen_correspondence.py"
    command = plan.commands[-1]
    assert command[command.index("--num_parts") + 1] == "3"
    assert command[command.index("--save_dir") + 1].endswith(
        "/runs/external_baseline_suite_v1/partnet_1"
    )
    assert "--no_wandb" in command
    assert plan.oracle_inputs == ("gt_part_count",)
    assert dict(plan.environment)["PYTHONPATH"].endswith("/DigitalTwinArt/mycuda")


def test_dta_plan_accepts_non_release_config(tmp_path: Path) -> None:
    plan = plan_dta_run(
        tmp_path / "DigitalTwinArt",
        _package(tmp_path),
        tmp_path / "result",
        object_id="partnet_1",
        gt_part_count=2,
        config_dir="config/smoke",
        no_wandb=False,
    )
    command = plan.commands[0]
    train_command = plan.commands[-1]
    assert train_command[train_command.index("--cfg_dir") + 1] == "config/smoke"
    assert "--no_wandb" not in train_command


def test_artgs_plan_contains_all_official_stages(tmp_path: Path) -> None:
    plan = plan_artgs_run(
        tmp_path / "ArtGS",
        _package(tmp_path),
        tmp_path / "result",
        object_id="partnet_1",
        gt_part_count=4,
    )
    assert [command[1] for command in plan.commands] == [
        "train_coarse.py",
        "train_predict.py",
        "train.py",
    ]
    assert plan.oracle_gt_part_count == 4
    assert plan.oracle_inputs == ("gt_part_count",)
    assert all("--source_path" in command for command in plan.commands)
    assert plan.commands[0][plan.commands[0].index("--model_path") + 1].endswith(
        "/outputs/external_suite/aligned/partnet_1/coarse_gs"
    )
    artgs_pythonpath = dict(plan.environment)["PYTHONPATH"]
    assert artgs_pythonpath.split(":")[0].endswith(
        "/ArtGS/submodules/diff-gaussian-rasterization"
    )


def test_register_artgs_slot_count_preserves_existing_scenes(tmp_path: Path) -> None:
    repo = tmp_path / "ArtGS"
    arguments = repo / "arguments"
    arguments.mkdir(parents=True)
    path = arguments / "num_slots.json"
    path.write_text(
        json.dumps({"external_suite": {"aligned": {"existing": 2}}}),
        encoding="utf-8",
    )

    register_artgs_slot_count(
        repo,
        dataset="external_suite",
        subset="aligned",
        object_id="partnet_10638",
        gt_part_count=3,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["external_suite"]["aligned"] == {
        "existing": 2,
        "partnet_10638": 3,
    }


def test_dry_run_writes_no_fake_metrics(tmp_path: Path) -> None:
    plan = plan_dta_run(
        tmp_path / "DigitalTwinArt",
        _package(tmp_path),
        tmp_path / "result",
        object_id="partnet_1",
        gt_part_count=2,
    )
    metrics = tmp_path / "metrics.json"
    result = execute_run_plan(
        plan, repo=tmp_path / "DigitalTwinArt", metrics_path=metrics, dry_run=True
    )
    saved = json.loads(metrics.read_text(encoding="utf-8"))
    assert result["status"] == "planned"
    assert saved["supported_metrics"] == {}
    assert saved["oracle"]["used_for_inference"] is True


def test_execute_run_plan_executes_with_declared_environment(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.json"
    plan = BaselineRunPlan(
        method="test",
        object_id="object",
        oracle_gt_part_count=None,
        commands=[["sh", "-c", 'test "$BASELINE_TEST_VALUE" = expected']],
        output_dir=str(tmp_path),
        environment=(("BASELINE_TEST_VALUE", "expected"),),
    )
    result = execute_run_plan(
        plan, repo=tmp_path, metrics_path=metrics, dry_run=False
    )
    assert result["status"] == "completed_unadapted"
    assert json.loads(metrics.read_text())["status"] == "completed_unadapted"


def test_execute_run_plan_recovers_only_when_declared_artifacts_exist(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "native_prediction.json"
    metrics = tmp_path / "metrics.json"
    plan = BaselineRunPlan(
        method="test",
        object_id="object",
        oracle_gt_part_count=None,
        commands=[["sh", "-c", f"touch {artifact!s}; exit 1"]],
        output_dir=str(tmp_path),
        completion_artifacts=(str(artifact),),
    )
    result = execute_run_plan(
        plan, repo=tmp_path, metrics_path=metrics, dry_run=False
    )
    assert result["status"] == "completed_with_exporter_failure"
    assert result["warning_stage"] == "official_process_teardown"


def test_execute_run_plan_keeps_failure_when_completion_artifact_is_missing(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "metrics.json"
    plan = BaselineRunPlan(
        method="test",
        object_id="object",
        oracle_gt_part_count=None,
        commands=[["sh", "-c", "exit 1"]],
        output_dir=str(tmp_path),
        completion_artifacts=(str(tmp_path / "missing.json"),),
    )
    result = execute_run_plan(
        plan, repo=tmp_path, metrics_path=metrics, dry_run=False
    )
    assert result["status"] == "failed"
    assert result["failure_stage"] == "official_inference"


def test_gaussianart_plan_explicitly_labels_oracles(tmp_path: Path) -> None:
    plan = plan_gaussianart_run(
        tmp_path / "GaussianArt",
        _package(tmp_path),
        tmp_path / "result",
        object_id="partnet_1",
        gt_part_count=4,
    )
    assert plan.commands[0][1] == "run.py"
    assert "part_semantic_initialization" in plan.oracle_inputs


def test_paris_and_ditto_reject_multi_part_objects(tmp_path: Path) -> None:
    package = _package(tmp_path)
    for planner, kwargs in (
        (plan_paris_run, {}),
        (plan_ditto_run, {"checkpoint": tmp_path / "ditto.ckpt"}),
    ):
        try:
            planner(
                tmp_path / planner.__name__,
                package,
                tmp_path / "result",
                object_id="partnet_1",
                gt_part_count=3,
                **kwargs,
            )
        except ValueError as exc:
            assert "supports only" in str(exc)
        else:
            raise AssertionError("multi-part input should be rejected")


def test_paris_non_oracle_plan_uses_se3_config(tmp_path: Path) -> None:
    plan = plan_paris_run(
        tmp_path / "PARIS",
        _package(tmp_path),
        tmp_path / "result",
        object_id="partnet_1",
        gt_part_count=2,
    )
    assert "configs/se3.yaml" in plan.commands[0]
    assert plan.oracle_gt_part_count is None
    assert "model.ray_chunk=1024" in plan.commands[0]
    assert any(
        argument.startswith("model.motion_gt_path=")
        and argument.endswith("/paris/textured_objs/trans.json")
        for argument in plan.commands[0]
    )
    assert dict(plan.environment)["TORCH_EXTENSIONS_DIR"].endswith(
        "/result/torch_extensions"
    )


def test_paris_plan_uses_exported_image_resolution(tmp_path: Path) -> None:
    package = _package(tmp_path)
    train = package / "paris" / "start" / "train"
    train.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (480, 352)).save(train / "0000.png")
    plan = plan_paris_run(
        tmp_path / "PARIS",
        package,
        tmp_path / "result",
        object_id="partnet_1",
        gt_part_count=2,
    )
    assert "dataset.img_wh=[480,352]" in plan.commands[0]


def test_videoartgs_plan_passes_root_for_internal_scene_join(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "videoartgs_data"
    scene = source_root / "external_suite" / "aligned" / "partnet_1"
    scene.mkdir(parents=True)
    for name in ("data.npz", "filtered.npz", "joint_infos.json"):
        (scene / name).write_bytes(b"data")

    plan = plan_videoartgs_run(
        tmp_path / "VideoArtGS",
        source_root,
        tmp_path / "result",
        object_id="partnet_1",
        joint_inventory_source="gt_oracle",
    )

    for command in plan.commands[:-1]:
        assert command[command.index("--source_path") + 1] == str(
            source_root.resolve()
        )
    render_command = plan.commands[-1]
    assert render_command[render_command.index("--source_path") + 1] == str(
        scene.resolve()
    )
    assert render_command[render_command.index("--iteration") + 1] == "20000"
    assert plan.commands[0][plan.commands[0].index("--model_path") + 1] == str(
        (scene / "init").resolve()
    )
    assert plan.commands[-1][plan.commands[-1].index("--model_path") + 1] == str(
        (scene / "final").resolve()
    )
    environment = dict(plan.environment)
    assert environment["PYTHONUTF8"] == "1"
    assert environment["TMPDIR"].endswith("/result/_build/tmp")
    assert environment["TORCH_EXTENSIONS_DIR"].endswith(
        "/result/_build/torch_extensions"
    )

    resumed = plan_videoartgs_run(
        tmp_path / "VideoArtGS",
        source_root,
        tmp_path / "result",
        object_id="partnet_1",
        joint_inventory_source="gt_oracle",
        start_stage=2,
    )
    assert len(resumed.commands) == 2
    assert resumed.commands[0][1] == "train.py"
    assert resumed.commands[1][1] == "render.py"

    training_only = plan_videoartgs_run(
        tmp_path / "VideoArtGS",
        source_root,
        tmp_path / "result",
        object_id="partnet_1",
        joint_inventory_source="gt_oracle",
        start_stage=2,
        include_reconstruction_export=False,
    )
    assert len(training_only.commands) == 1
