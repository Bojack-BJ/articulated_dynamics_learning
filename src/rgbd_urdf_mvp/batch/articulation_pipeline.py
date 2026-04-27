from __future__ import annotations

import copy
import os
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

from ..core.categories import normalize_category
from ..core.cli_config import config_to_argv, load_command_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

DEFAULT_RECORD_CONFIG_BY_CATEGORY = {
    "microwave": PROJECT_ROOT / "configs" / "record_microwave.yaml",
    "refrigerator": PROJECT_ROOT / "configs" / "record_refrigerator.yaml",
    "door": PROJECT_ROOT / "configs" / "record_hinge.yaml",
    "oven": PROJECT_ROOT / "configs" / "record_hinge.yaml",
    "dishwasher": PROJECT_ROOT / "configs" / "record_hinge.yaml",
    "toasteroven": PROJECT_ROOT / "configs" / "record_hinge.yaml",
    "electrickettle": PROJECT_ROOT / "configs" / "record_hinge.yaml",
    "drawer": PROJECT_ROOT / "configs" / "record_drawer.yaml",
}
DEFAULT_TRACK_CONFIG = PROJECT_ROOT / "configs" / "track_default.yaml"

PREFERRED_JOINT_TOKENS = {
    "microwave": ("micro", "door", "hinge"),
    "refrigerator": ("door", "freezer"),
    "drawer": ("drawer", "slide"),
    "door": ("door", "hinge"),
}


@dataclass(frozen=True)
class BatchObjectSpec:
    category: str
    model_path: Path
    object_id: str
    joint_name: str


@dataclass(frozen=True)
class BatchObjectArtifacts:
    episode_dir: Path
    episode_json: Path
    pointcloud_dir: Path
    manifest_json: Path
    tracks_json: Path
    poses_json: Path
    joints_json: Path
    viewer_html: Path

    @classmethod
    def for_object(cls, output_root: Path, object_id: str) -> "BatchObjectArtifacts":
        episode_dir = output_root / object_id
        pointcloud_dir = episode_dir / "pointcloud_4d_partseg"
        return cls(
            episode_dir=episode_dir,
            episode_json=episode_dir / "episode.json",
            pointcloud_dir=pointcloud_dir,
            manifest_json=pointcloud_dir / "fusion_manifest.json",
            tracks_json=pointcloud_dir / "part_tracks.json",
            poses_json=pointcloud_dir / "part_poses.json",
            joints_json=pointcloud_dir / "joint_inference.json",
            viewer_html=pointcloud_dir / "viewer_pose_flow.html",
        )


@dataclass(frozen=True)
class ArticulationBatchConfig:
    manifest_path: Path
    resume: bool = False
    skip_existing: bool = False
    jobs: int = 1
    output_root: Path = PROJECT_ROOT / "outputs" / "recordings"
    batch_log_dir: Path = PROJECT_ROOT / "outputs" / "recordings" / "_batch_logs"
    torch_home: Path = PROJECT_ROOT / ".cache" / "torch"
    record_config: Path | None = None
    track_config: Path | None = None
    cotracker_repo: Path | None = None
    cotracker_checkpoint: Path | None = None
    track_device: str | None = None
    tracking_jobs: int | None = None
    fuse_pixel_stride: int = 8
    fuse_voxel_size_m: float = 0.02
    min_tracks_per_part: int = 4
    mujoco_prior_mode: str = "off"
    generate_viewer: bool = True


class ArticulationBatchRunner:
    def __init__(self, config: ArticulationBatchConfig) -> None:
        self.config = config
        self._console_lock = threading.Lock()
        self._manifest_path = config.manifest_path.expanduser().resolve()
        self._output_root = resolve_project_path(config.output_root)
        self._batch_log_dir = resolve_project_path(config.batch_log_dir)
        self._torch_home = resolve_project_path(config.torch_home)
        effective_track_device = (config.track_device or self._template_track_device() or "auto").lower()
        default_tracking_jobs = 1 if effective_track_device in {"auto", "mps", "cuda"} else max(1, config.jobs)
        self._tracking_jobs = max(1, int(config.tracking_jobs or default_tracking_jobs))
        self._tracking_semaphore = threading.Semaphore(self._tracking_jobs)

    def run(self) -> dict[str, Any]:
        specs = parse_batch_manifest(self._manifest_path)
        total = len(specs)
        failures: list[str] = []
        completed = 0
        skipped = 0

        if self.config.jobs <= 1:
            self._console(
                f"Queued {total} objects from {self._manifest_path} "
                f"(jobs=1, tracking_jobs={self._tracking_jobs}, output_root={self._output_root})"
            )
            for index, spec in enumerate(specs, start=1):
                result = self._process_object(spec, index=index, total=total, stream=sys.stdout)
                if result == "skipped":
                    skipped += 1
                else:
                    completed += 1
            return {
                "manifest_path": str(self._manifest_path),
                "objects_total": len(specs),
                "objects_completed": completed,
                "objects_skipped": skipped,
                "objects_failed": failures,
                "jobs": self.config.jobs,
            }

        self._batch_log_dir.mkdir(parents=True, exist_ok=True)
        self._console(
            f"Queued {total} objects from {self._manifest_path} "
            f"(jobs={self.config.jobs}, tracking_jobs={self._tracking_jobs}, logs={self._batch_log_dir})"
        )
        with ThreadPoolExecutor(max_workers=self.config.jobs) as executor:
            future_map = {}
            for index, spec in enumerate(specs, start=1):
                log_path = self._batch_log_dir / f"{spec.object_id}.log"
                future = executor.submit(self._process_object_with_log, spec, index, total, log_path)
                future_map[future] = (spec, index, total, log_path)
            for future in as_completed(future_map):
                spec, index, total, log_path = future_map[future]
                try:
                    result = future.result()
                except Exception:
                    failures.append(spec.object_id)
                    self._console(
                        f"[{spec.object_id} {index}/{total}] failed; log={log_path}",
                        error=True,
                    )
                    continue
                if result == "skipped":
                    skipped += 1
                    self._console(
                        f"[{spec.object_id} {index}/{total}] skipped; log={log_path}"
                    )
                else:
                    completed += 1
                    self._console(
                        f"[{spec.object_id} {index}/{total}] completed; log={log_path}"
                    )

        if failures:
            raise RuntimeError(f"Batch pipeline finished with failures: {' '.join(failures)}")
        return {
            "manifest_path": str(self._manifest_path),
            "objects_total": len(specs),
            "objects_completed": completed,
            "objects_skipped": skipped,
            "objects_failed": failures,
            "jobs": self.config.jobs,
            "batch_log_dir": str(self._batch_log_dir),
        }

    def _process_object_with_log(self, spec: BatchObjectSpec, index: int, total: int, log_path: Path) -> str:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as stream:
            return self._process_object(spec, index=index, total=total, stream=stream)

    def _process_object(self, spec: BatchObjectSpec, index: int, total: int, stream: TextIO) -> str:
        artifacts = BatchObjectArtifacts.for_object(self._output_root, spec.object_id)
        terminal_artifact = artifacts.viewer_html if self.config.generate_viewer else artifacts.joints_json
        if self.config.skip_existing and terminal_artifact.exists():
            self._announce(
                spec.object_id,
                index,
                total,
                f"skip-existing: found {terminal_artifact}",
                stream,
                to_console=self.config.jobs > 1,
            )
            return "skipped"

        self._announce(
            spec.object_id,
            index,
            total,
            f"started; log={self._batch_log_dir / f'{spec.object_id}.log'}" if self.config.jobs > 1 else "started",
            stream,
            to_console=self.config.jobs > 1,
        )
        model_path = self._resolve_model_xml_path(spec.category, spec.model_path, stream)
        joint_name = infer_joint_name(spec.category, model_path, spec.joint_name)

        if self._should_skip(artifacts.episode_json):
            self._announce(
                spec.object_id,
                index,
                total,
                f"resume: skip record-mujoco; found {artifacts.episode_json}",
                stream,
            )
        else:
            self._announce(spec.object_id, index, total, f"record-mujoco joint={joint_name}", stream)
            record_argv = build_cli_argv_from_template(
                self._resolve_record_config_path(spec.category),
                {
                    "model": model_path,
                    "category": spec.category,
                    "object-id": spec.object_id,
                    "output-dir": self._output_root,
                    "joint-name": joint_name,
                },
            )
            self._run_cli(record_argv, stream)

        if self._should_skip(artifacts.manifest_json):
            self._announce(
                spec.object_id,
                index,
                total,
                f"resume: skip fuse-pointcloud; found {artifacts.manifest_json}",
                stream,
            )
        else:
            self._announce(spec.object_id, index, total, "fuse-pointcloud", stream)
            self._run_cli(
                [
                    "fuse-pointcloud",
                    str(artifacts.episode_json),
                    "--output-dir",
                    str(artifacts.pointcloud_dir),
                    "--pixel-stride",
                    str(max(1, int(self.config.fuse_pixel_stride))),
                    "--voxel-size-m",
                    str(float(self.config.fuse_voxel_size_m)),
                ],
                stream,
            )

        if self._should_skip(artifacts.tracks_json):
            self._announce(
                spec.object_id,
                index,
                total,
                f"resume: skip track-part-pixels; found {artifacts.tracks_json}",
                stream,
            )
        else:
            self._validate_tracking_inputs()
            self._announce(
                spec.object_id,
                index,
                total,
                f"waiting for tracking slot ({self._tracking_jobs} max)",
                stream,
            )
            with self._tracking_semaphore:
                self._announce(spec.object_id, index, total, "track-part-pixels", stream)
                track_overrides: dict[str, Any] = {
                    "episode": artifacts.episode_json,
                    "output-json": artifacts.tracks_json,
                }
                if self.config.cotracker_repo is not None:
                    track_overrides["cotracker-repo"] = resolve_project_path(self.config.cotracker_repo)
                if self.config.cotracker_checkpoint is not None:
                    track_overrides["cotracker-checkpoint"] = resolve_project_path(self.config.cotracker_checkpoint)
                if self.config.track_device is not None:
                    track_overrides["device"] = self.config.track_device
                track_argv = build_cli_argv_from_template(
                    self._resolve_track_config_path(),
                    track_overrides,
                )
                self._run_cli(track_argv, stream)

        if self._should_skip(artifacts.poses_json):
            self._announce(
                spec.object_id,
                index,
                total,
                f"resume: skip estimate-part-poses; found {artifacts.poses_json}",
                stream,
            )
        else:
            self._announce(spec.object_id, index, total, "estimate-part-poses", stream)
            self._run_cli(
                [
                    "estimate-part-poses",
                    str(artifacts.tracks_json),
                    "--method",
                    "tracks",
                    "--output-json",
                    str(artifacts.poses_json),
                    "--min-tracks-per-part",
                    str(max(3, int(self.config.min_tracks_per_part))),
                ],
                stream,
            )

        if self._should_skip(artifacts.joints_json):
            self._announce(
                spec.object_id,
                index,
                total,
                f"resume: skip infer-joints; found {artifacts.joints_json}",
                stream,
            )
        else:
            self._announce(spec.object_id, index, total, "infer-joints", stream)
            self._run_cli(
                [
                    "infer-joints",
                    str(artifacts.poses_json),
                    "--output-json",
                    str(artifacts.joints_json),
                    "--mujoco-prior",
                    self.config.mujoco_prior_mode,
                ],
                stream,
            )

        if self.config.generate_viewer:
            if self._should_skip(artifacts.viewer_html):
                self._announce(
                    spec.object_id,
                    index,
                    total,
                    f"resume: skip visualize-pointcloud; found {artifacts.viewer_html}",
                    stream,
                )
            else:
                self._announce(spec.object_id, index, total, "visualize-pointcloud", stream)
                self._run_cli(
                    [
                        "visualize-pointcloud",
                        str(artifacts.manifest_json),
                        "--output-html",
                        str(artifacts.viewer_html),
                        "--part-tracks-json",
                        str(artifacts.tracks_json),
                        "--part-poses-json",
                        str(artifacts.poses_json),
                        "--joint-inference-json",
                        str(artifacts.joints_json),
                    ],
                    stream,
                )
        return "completed"

    def _should_skip(self, output_path: Path) -> bool:
        return bool(self.config.resume and output_path.exists())

    def _resolve_record_config_path(self, category: str) -> Path:
        if self.config.record_config is not None:
            return resolve_project_path(self.config.record_config)
        config_path = DEFAULT_RECORD_CONFIG_BY_CATEGORY.get(category)
        if config_path is None:
            raise ValueError(
                f"No recording preset for category '{category}'. "
                "Add a YAML preset under configs/ and map it in DEFAULT_RECORD_CONFIG_BY_CATEGORY."
            )
        return config_path

    def _resolve_track_config_path(self) -> Path:
        if self.config.track_config is not None:
            return resolve_project_path(self.config.track_config)
        return DEFAULT_TRACK_CONFIG

    def _template_track_device(self) -> str | None:
        config_path = self._resolve_track_config_path()
        if not config_path.exists():
            return None
        payload = load_command_config(config_path)
        args = payload.get("args", {})
        if not isinstance(args, dict):
            return None
        value = args.get("device")
        return str(value) if value is not None else None

    def _resolve_model_xml_path(self, category: str, model_path: Path, stream: TextIO) -> Path:
        resolved = resolve_project_path(model_path)
        suffix = resolved.suffix.lower()
        if suffix == ".xml":
            return resolved
        if suffix != ".usd":
            raise ValueError(f"Unsupported model path for batch pipeline: {resolved}")
        stem = resolved.stem
        output_prefix = PROJECT_ROOT / "examples" / "mujoco_models" / stem
        output_xml = output_prefix.with_suffix(".xml")
        if not output_xml.exists():
            self._log(stream, f"=== [{stem}] usd_to_mjcf ===")
            self._run_module(
                "rgbd_urdf_mvp.sim.usd_to_mjcf",
                [
                    str(resolved),
                    "--category",
                    category,
                    "--output-prefix",
                    str(output_prefix),
                ],
                stream,
            )
        return output_xml

    def _validate_tracking_inputs(self) -> None:
        track_config_path = self._resolve_track_config_path()
        if not track_config_path.exists():
            raise FileNotFoundError(f"Track config does not exist: {track_config_path}")
        if self.config.cotracker_repo is not None and not resolve_project_path(self.config.cotracker_repo).is_dir():
            raise FileNotFoundError(f"CoTracker repo does not exist: {self.config.cotracker_repo}")
        if self.config.cotracker_checkpoint is not None and not resolve_project_path(self.config.cotracker_checkpoint).is_file():
            raise FileNotFoundError(f"CoTracker checkpoint does not exist: {self.config.cotracker_checkpoint}")

    def _run_cli(self, argv: list[str], stream: TextIO) -> None:
        command = [sys.executable, "-m", "rgbd_urdf_mvp", *argv]
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=self._subprocess_env(),
            stdout=stream,
            stderr=stream,
            text=True,
            check=True,
        )

    def _run_module(self, module: str, argv: list[str], stream: TextIO) -> None:
        command = [sys.executable, "-m", module, *argv]
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=self._subprocess_env(),
            stdout=stream,
            stderr=stream,
            text=True,
            check=True,
        )

    def _subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        pythonpath_entries = [str(SRC_ROOT)]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        env["TORCH_HOME"] = str(self._torch_home)
        return env

    def _log(self, stream: TextIO, message: str) -> None:
        print(message, file=stream, flush=True)

    def _announce(
        self,
        object_id: str,
        index: int,
        total: int,
        message: str,
        stream: TextIO,
        *,
        to_console: bool = True,
    ) -> None:
        line = f"[{object_id} {index}/{total}] {message}"
        self._log(stream, line)
        if to_console:
            self._console(line)

    def _console(self, message: str, error: bool = False) -> None:
        with self._console_lock:
            print(message, file=sys.stderr if error else sys.stdout, flush=True)


def parse_batch_manifest(path: Path) -> list[BatchObjectSpec]:
    manifest_path = path.expanduser().resolve()
    specs: list[BatchObjectSpec] = []
    for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = raw_line.split("\t")
        if len(parts) != 4:
            raise ValueError(
                f"Manifest row {line_number} must contain exactly four tab-separated columns: "
                "category, model_path, object_id, joint_name."
            )
        category_text, model_text, object_id, joint_name = (part.strip() for part in parts)
        if not category_text or not model_text or not object_id or not joint_name:
            raise ValueError(f"Manifest row {line_number} is incomplete: {raw_line}")
        specs.append(
            BatchObjectSpec(
                category=normalize_category(category_text),
                model_path=Path(model_text),
                object_id=object_id,
                joint_name=joint_name,
            )
        )
    return specs


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def build_cli_argv_from_template(config_path: Path, overrides: Mapping[str, Any]) -> list[str]:
    from ..cli import build_parser

    payload = copy.deepcopy(load_command_config(resolve_project_path(config_path)))
    base_args = payload.get("args", {})
    if base_args is None:
        base_args = {}
    if not isinstance(base_args, dict):
        raise ValueError(f"Config field 'args' must be a mapping in {config_path}")
    merged_args = dict(base_args)
    for key, value in payload.items():
        if key not in {"command", "args", "description"}:
            merged_args.setdefault(key, value)
    merged_args.update(overrides)
    payload["args"] = merged_args
    return config_to_argv(payload, build_parser())


def infer_joint_name(category: str, model_xml: Path, requested_joint_name: str) -> str:
    if requested_joint_name and requested_joint_name != "auto":
        return requested_joint_name
    root = ET.parse(model_xml).getroot()
    joints: list[tuple[str, str]] = []
    for joint in root.findall(".//joint"):
        name = (joint.get("name") or "").strip()
        joint_type = (joint.get("type") or "hinge").strip().lower()
        if not name:
            continue
        joints.append((name, joint_type))
    if not joints:
        raise ValueError(f"No named joints found in {model_xml}")
    for token in PREFERRED_JOINT_TOKENS.get(category, ()):
        for name, _ in joints:
            if token in name.lower():
                return name
    for preferred_type in ("hinge", "slide"):
        for name, joint_type in joints:
            if joint_type == preferred_type:
                return name
    return joints[0][0]
