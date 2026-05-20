from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.serialization import load_json, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class ParticulateInferenceConfig:
    mesh_path: str | Path | None = None
    reconstruction_artifact_path: str | Path | None = None
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "particulate"
    particulate_root: str | Path = PROJECT_ROOT / "Particulate"
    python_bin: str | Path = os.environ.get("PARTICULATE_PYTHON", "python")
    model_config: str | Path = "configs/particulate-B.yaml"
    ckpt_path: str | Path | None = None
    up_dir: str = "-Z"
    num_points: int = 102400
    num_points_global: int = 40000
    target_faces: int | None = None
    min_part_confidence: float = 0.0
    strict: bool = True
    animation_frames: int = 50
    export_urdf: bool = True
    export_mjcf: bool = True
    eval: bool = True
    dry_run: bool = False


class ParticulateInferenceRunner:
    def run(self, config: ParticulateInferenceConfig) -> Path:
        particulate_root = _resolve_path(config.particulate_root)
        if not particulate_root.exists():
            raise FileNotFoundError(
                f"Particulate submodule not found at {particulate_root}. "
                "Run: git submodule update --init --recursive Particulate"
            )

        output_dir = _resolve_path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        source_mesh_path = self._resolve_mesh_path(config)
        mesh_path = self._prepare_mesh_path(source_mesh_path, output_dir, config.target_faces)

        command = [
            str(config.python_bin),
            "-m",
            "rgbd_urdf_mvp.perception.particulate_infer_wrapper",
            "--input_mesh",
            str(mesh_path),
            "--output_dir",
            str(output_dir),
            "--model_config",
            str(config.model_config),
            f"--up_dir={config.up_dir}",
            "--num_points",
            str(max(1, int(config.num_points))),
            "--num_points_global",
            str(max(1, int(config.num_points_global))),
            "--min_part_confidence",
            str(float(config.min_part_confidence)),
            "--animation_frames",
            str(max(2, int(config.animation_frames))),
        ]
        if config.ckpt_path is not None:
            command.extend(["--ckpt_path", str(_resolve_path(config.ckpt_path))])
        if not config.strict:
            command.append("--no_strict")
        if config.export_urdf:
            command.append("--export_urdf")
        if config.export_mjcf:
            command.append("--export_mjcf")
        if config.eval:
            command.append("--eval")

        manifest_path = output_dir / "particulate_result.json"
        manifest: dict[str, Any] = {
            "source": "particulate-submodule",
            "particulate_root": str(particulate_root),
            "input_mesh_path": str(source_mesh_path),
            "particulate_input_mesh_path": str(mesh_path),
            "output_dir": str(output_dir),
            "command": command,
            "dry_run": bool(config.dry_run),
            "artifacts": {},
        }

        if not config.dry_run:
            stdout_path = output_dir / "particulate_stdout.log"
            stderr_path = output_dir / "particulate_stderr.log"
            with stdout_path.open("w", encoding="utf-8") as stdout_file:
                with stderr_path.open("w", encoding="utf-8") as stderr_file:
                    subprocess.run(
                        command,
                        cwd=particulate_root,
                        env=self._subprocess_env(particulate_root),
                        text=True,
                        stdout=stdout_file,
                        stderr=stderr_file,
                        check=True,
                    )
            manifest["artifacts"] = self._collect_artifacts(output_dir)
            manifest["logs"] = {
                "stdout": str(stdout_path.resolve()),
                "stderr": str(stderr_path.resolve()),
            }
            if not any(manifest["artifacts"].values()):
                save_json(manifest, manifest_path)
                tail = _tail_text(stdout_path) + _tail_text(stderr_path)
                raise RuntimeError(
                    "PARTICULATE finished without producing artifacts. "
                    f"See logs: {stdout_path} {stderr_path}{tail}"
                )

        save_json(manifest, manifest_path)
        return manifest_path

    def _prepare_mesh_path(self, mesh_path: Path, output_dir: Path, target_faces: int | None) -> Path:
        if target_faces is None or int(target_faces) <= 0:
            return mesh_path

        try:
            import trimesh
        except ModuleNotFoundError as exc:
            raise RuntimeError("Mesh decimation requires trimesh in the wrapper environment") from exc

        mesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected a trimesh.Trimesh after loading {mesh_path}, got {type(mesh).__name__}")
        if len(mesh.faces) <= int(target_faces):
            return mesh_path

        simplified = self._simplify_mesh(mesh, int(target_faces))
        decimated_path = output_dir / f"{mesh_path.stem}.particulate_decimated.glb"
        simplified.export(decimated_path)
        return decimated_path

    def _simplify_mesh(self, mesh: Any, target_faces: int) -> Any:
        try:
            return mesh.simplify_quadric_decimation(face_count=target_faces)
        except TypeError:
            try:
                return mesh.simplify_quadric_decimation(target_faces)
            except Exception as exc:
                raise RuntimeError(
                    "Mesh decimation failed. Install fast-simplification/open3d in the wrapper environment, "
                    "or lower Hunyuan --face-count before running PARTICULATE."
                ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Mesh decimation failed. Install fast-simplification/open3d in the wrapper environment, "
                "or lower Hunyuan --face-count before running PARTICULATE."
            ) from exc

    def _resolve_mesh_path(self, config: ParticulateInferenceConfig) -> Path:
        if config.mesh_path is not None:
            return _resolve_path(config.mesh_path)
        if config.reconstruction_artifact_path is None:
            raise ValueError("Provide either mesh_path or reconstruction_artifact_path.")

        artifact_path = _resolve_path(config.reconstruction_artifact_path)
        artifact = load_json(artifact_path)
        mesh_text = artifact.get("canonical_mesh_path")
        if not isinstance(mesh_text, str) or not mesh_text:
            raise ValueError(f"Reconstruction artifact has no canonical_mesh_path: {artifact_path}")
        return _resolve_path(mesh_text)

    def _collect_artifacts(self, output_dir: Path) -> dict[str, Any]:
        animated = sorted(output_dir.glob("animated_textured*.glb"))
        segmented = sorted(output_dir.glob("mesh_parts_with_axes*.glb"))
        urdfs = sorted(output_dir.glob("urdf_*/model.urdf"))
        mjcfs = sorted(output_dir.glob("mjcf_*/model.xml"))
        eval_npz = output_dir / "eval" / "pred.npz"
        eval_obj = output_dir / "eval" / "pred.obj"
        return {
            "animated_glb": str(animated[-1].resolve()) if animated else None,
            "mesh_parts_glb": str(segmented[-1].resolve()) if segmented else None,
            "urdf": str(urdfs[-1].resolve()) if urdfs else None,
            "mjcf": str(mjcfs[-1].resolve()) if mjcfs else None,
            "eval_npz": str(eval_npz.resolve()) if eval_npz.exists() else None,
            "eval_obj": str(eval_obj.resolve()) if eval_obj.exists() else None,
        }

    def _subprocess_env(self, particulate_root: Path) -> dict[str, str]:
        env = dict(os.environ)
        pythonpath_entries = [str(PROJECT_ROOT / "src"), str(particulate_root), str(particulate_root / "PartField")]
        if env.get("PYTHONPATH"):
            pythonpath_entries.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        return env


@dataclass(frozen=True)
class ArticulationBackendComparisonConfig:
    tracking_joint_inference_path: str | Path
    particulate_result_path: str | Path
    output_json: str | Path | None = None


class ArticulationBackendComparator:
    def compare(self, config: ArticulationBackendComparisonConfig) -> Path:
        tracking_path = _resolve_path(config.tracking_joint_inference_path)
        particulate_path = _resolve_path(config.particulate_result_path)
        output_path = (
            _resolve_path(config.output_json)
            if config.output_json is not None
            else particulate_path.with_name("articulation_backend_comparison.json")
        )

        tracking = load_json(tracking_path)
        particulate = load_json(particulate_path)
        summary = {
            "tracking": self._summarize_tracking(tracking, tracking_path),
            "particulate": self._summarize_particulate(particulate, particulate_path),
            "notes": [
                "This comparison is a structural summary, not a geometric alignment metric.",
                "Tracking estimates motion from observed RGB-D trajectories; PARTICULATE predicts articulation from a canonical mesh.",
            ],
        }
        save_json(summary, output_path)
        return output_path

    def _summarize_tracking(self, payload: dict[str, Any], path: Path) -> dict[str, Any]:
        joints = payload.get("joints", [])
        if not isinstance(joints, list):
            joints = []
        type_counts: dict[str, int] = {}
        for joint in joints:
            if not isinstance(joint, dict):
                continue
            joint_type = str(joint.get("joint_type", "unknown"))
            type_counts[joint_type] = type_counts.get(joint_type, 0) + 1
        return {
            "source_path": str(path),
            "estimator": payload.get("estimator"),
            "joint_count": len(joints),
            "joint_type_counts": type_counts,
            "anchor_part_id": payload.get("anchor_part_id"),
            "joints": [
                {
                    "name": joint.get("name"),
                    "joint_type": joint.get("joint_type"),
                    "parent": joint.get("parent_name"),
                    "child": joint.get("child_name"),
                    "axis": joint.get("axis"),
                    "pivot": joint.get("pivot"),
                    "limits": joint.get("limits"),
                    "confidence": joint.get("confidence"),
                }
                for joint in joints
                if isinstance(joint, dict)
            ],
        }

    def _summarize_particulate(self, payload: dict[str, Any], path: Path) -> dict[str, Any]:
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        eval_npz = artifacts.get("eval_npz")
        summary: dict[str, Any] = {
            "source_path": str(path),
            "input_mesh_path": payload.get("input_mesh_path"),
            "output_dir": payload.get("output_dir"),
            "artifacts": artifacts,
        }
        if isinstance(eval_npz, str) and Path(eval_npz).exists():
            summary.update(self._summarize_particulate_npz(Path(eval_npz)))
        return summary

    def _summarize_particulate_npz(self, path: Path) -> dict[str, Any]:
        import numpy as np

        data = np.load(path, allow_pickle=True)
        face_part_ids = data["face_part_ids"] if "face_part_ids" in data.files else None
        motion_hierarchy = data["motion_hierarchy"] if "motion_hierarchy" in data.files else None
        is_revolute = data["is_part_revolute"] if "is_part_revolute" in data.files else None
        is_prismatic = data["is_part_prismatic"] if "is_part_prismatic" in data.files else None
        return {
            "eval_npz_path": str(path),
            "part_count": int(len(np.unique(face_part_ids))) if face_part_ids is not None else None,
            "hierarchy_edge_count": int(len(motion_hierarchy)) if motion_hierarchy is not None else None,
            "revolute_part_count": int(np.asarray(is_revolute).sum()) if is_revolute is not None else None,
            "prismatic_part_count": int(np.asarray(is_prismatic).sum()) if is_prismatic is not None else None,
        }


def _resolve_path(path: str | Path | None) -> Path:
    if path is None:
        raise ValueError("Path must not be None")
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text:
        return ""
    return f"\n--- tail {path.name} ---\n{text[-max_chars:]}"
