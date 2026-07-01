from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Callable, Literal
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..core.serialization import load_json, save_json
from .ballistic import load_model, resolve_joint, resolve_model_path


@dataclass(slots=True)
class ILRolloutRenderConfig:
    dataset_path: str | Path
    mjcf_path: str | Path
    output_path: str | Path | None = None
    robot_model_path: str | Path | None = None
    dynamics_artifact_path: str | Path | None = None
    episode_index: int = 0
    trajectory_source: Literal["dataset", "resimulate"] = "dataset"
    joint_name: str | None = None
    joint_id: int | None = None
    sim_dt: float | None = None
    width: int = 960
    height: int = 720
    fps: int = 30
    camera_azimuth: float = 135.0
    camera_elevation: float = -25.0
    camera_distance: float | None = None
    camera_name: str | None = None
    object_visual_scale: float = 1.0
    playback_slowdown: float = 1.0
    robot_base_pos: tuple[float, float, float] = (-0.35, -0.65, 0.0)
    robot_base_euler: tuple[float, float, float] = (0.0, 0.0, 0.8)
    robot_motion: Literal["static", "ik-pulse"] = "ik-pulse"
    robot_ee_body: str = "x5_link8"
    robot_approach_distance: float = 0.08
    robot_pulse_distance: float = 0.06
    dynamics_body_name: str | None = None
    dynamics_joint_name: str | None = None
    enable_contact: bool = False
    avoid_robot_self_collision: bool = True
    self_collision_penetration_tolerance_m: float = 0.0
    avoid_robot_object_collision: bool = True
    object_collision_penetration_tolerance_m: float = 0.0
    gravity_mode: str = "zero"
    write_frames: bool = False


class ILRolloutRenderer:
    """Render one ballistic/contact IL demo as a MuJoCo animation."""

    def render(self, config: ILRolloutRenderConfig) -> Path:
        dataset_path = Path(config.dataset_path).expanduser().resolve()
        data = np.load(dataset_path)
        obs = np.asarray(data["obs"], dtype=float)
        action = np.asarray(data["action"], dtype=float)
        response = np.asarray(data["free_response"], dtype=float)
        episode_index = int(np.clip(int(config.episode_index), 0, obs.shape[0] - 1))
        output_path = self._output_path(dataset_path, config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_robot_trace: list[dict[str, object]] = []

        render_mjcf = self._prepare_render_mjcf(config, output_path.parent)
        model = load_model(
            render_mjcf,
            sim_dt=config.sim_dt,
            enable_contact=bool(config.enable_contact),
            gravity_mode=str(config.gravity_mode),
        )
        joint = resolve_joint(model, joint_name=config.joint_name, joint_id=config.joint_id)
        self._apply_dynamics_artifact(model, joint, config)
        duration_s = max(float(obs[episode_index, 3]), float(model.opt.timestep))
        slowdown = max(1.0, float(config.playback_slowdown))
        frame_count = max(2, int(round(duration_s * max(1, int(config.fps)) * slowdown)) + 1)
        frame_times = np.linspace(0.0, duration_s, frame_count)

        frames = self._render_frames(
            model=model,
            joint=joint,
            obs_row=obs[episode_index],
            action_row=action[episode_index],
            response=response[episode_index],
            frame_times=frame_times,
            config=config,
        )
        if output_path.suffix.lower() not in {"", ".dir", ".mp4", ".gif"}:
            output_path = output_path.with_suffix(".mp4")
        if config.write_frames or output_path.suffix.lower() in {"", ".dir"}:
            frame_dir = output_path if output_path.suffix.lower() in {"", ".dir"} else output_path.with_suffix("")
            self._write_frame_sequence(frame_dir, frames)
            self._write_ik_debug(frame_dir.with_suffix(".ik_debug.json"), config)
            return frame_dir
        self._write_video(output_path, frames, int(config.fps))
        self._write_ik_debug(output_path.with_suffix(".ik_debug.json"), config)
        return output_path

    def _prepare_render_mjcf(self, config: ILRolloutRenderConfig, output_dir: Path) -> Path:
        object_path = resolve_model_path(config.mjcf_path)
        if config.robot_model_path is None:
            return object_path
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_path = output_dir / f"{object_path.stem}.with_x5_scene.xml"
        _write_combined_object_robot_mjcf(
            object_mjcf=object_path,
            robot_model=Path(config.robot_model_path).expanduser().resolve(),
            output_path=combined_path,
            robot_base_pos=config.robot_base_pos,
            robot_base_euler=config.robot_base_euler,
            object_visual_scale=max(1e-6, float(config.object_visual_scale)),
        )
        return combined_path

    def _apply_dynamics_artifact(self, model, joint, config: ILRolloutRenderConfig) -> None:
        if config.dynamics_artifact_path is None:
            return
        artifact = load_json(Path(config.dynamics_artifact_path).expanduser().resolve())
        params = artifact.get("identified_parameters", {})
        part_rows = params.get("parts", []) if isinstance(params, dict) else []
        joint_rows = params.get("joints", []) if isinstance(params, dict) else []
        body_name = config.dynamics_body_name
        if body_name is None:
            optimized = [row for row in part_rows if isinstance(row, dict) and row.get("optimized")]
            if len(optimized) == 1:
                body_name = str(optimized[0].get("name", ""))
        for row in part_rows:
            if not isinstance(row, dict):
                continue
            if body_name is not None and str(row.get("name", "")) != body_name:
                continue
            body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(row.get("name", ""))))
            if body_id < 0:
                continue
            if "mass_kg" in row:
                model.body_mass[body_id] = max(1e-9, float(row["mass_kg"]))
            if "inertia_diag" in row and len(row["inertia_diag"]) == 3:
                model.body_inertia[body_id] = np.maximum(1e-9, np.asarray(row["inertia_diag"], dtype=float))
        target_joint_name = config.dynamics_joint_name or config.joint_name
        for row in joint_rows:
            if not isinstance(row, dict):
                continue
            row_name = str(row.get("name", ""))
            if target_joint_name is not None and row_name != target_joint_name:
                if len(joint_rows) != 1:
                    continue
            dof = joint.dof_adr
            if "damping" in row:
                model.dof_damping[dof] = max(0.0, float(row["damping"]))
            if "frictionloss" in row:
                model.dof_frictionloss[dof] = max(0.0, float(row["frictionloss"]))
        try:
            mujoco.mj_setConst(model, mujoco.MjData(model))
        except Exception:
            pass

    def _render_frames(
        self,
        model,
        joint,
        obs_row: np.ndarray,
        action_row: np.ndarray,
        response: np.ndarray,
        frame_times: np.ndarray,
        config: ILRolloutRenderConfig,
    ) -> list[np.ndarray]:
        try:
            renderer = mujoco.Renderer(model, width=max(64, int(config.width)), height=max(64, int(config.height)))
        except Exception as exc:
            raise RuntimeError(
                "MuJoCo renderer could not create an OpenGL context. On macOS this often means the "
                "command is running in a sandbox or headless session without a CoreGraphics connection. "
                "Run the same render-il-rollout command from a normal local terminal, or use an approved "
                "unsandboxed Codex command."
            ) from exc
        data = mujoco.MjData(model)
        q0 = float(obs_row[0])
        target_q = float(obs_row[2])
        action_kind = "contact" if action_row.shape[0] >= 8 else "initial_qvel"
        robot_ik = _RobotIKState.create(model, config.robot_ee_body) if config.robot_model_path and config.robot_motion == "ik-pulse" else None
        frames: list[np.ndarray] = []
        if config.trajectory_source == "resimulate":
            states = self._resimulated_states(
                model,
                joint,
                q0,
                action_row,
                frame_times,
                object_visual_scale=max(1e-6, float(config.object_visual_scale)),
            )
        elif config.trajectory_source == "dataset":
            states = self._dataset_states(response, frame_times, action_row)
        else:
            raise ValueError(f"Unsupported trajectory_source: {config.trajectory_source!r}")

        for index, (time_s, q, qdot, pulse_active) in enumerate(states):
            data.time = float(time_s)
            data.qpos[joint.qpos_adr] = float(np.clip(q, joint.lower, joint.upper))
            data.qvel[joint.dof_adr] = float(qdot)
            mujoco.mj_forward(model, data)
            if robot_ik is not None and action_row.shape[0] >= 8:
                ee_target = _robot_pulse_target(
                    action_row,
                    float(time_s),
                    approach_distance=float(config.robot_approach_distance),
                    pulse_distance=float(config.robot_pulse_distance),
                    point_scale=max(1e-6, float(config.object_visual_scale)),
                )
                collision_ok = None
                if config.enable_contact and (config.avoid_robot_self_collision or config.avoid_robot_object_collision):
                    collision_ok = lambda candidate_data: (
                        (
                            not config.avoid_robot_self_collision
                            or not _has_robot_self_penetration(
                                model,
                                candidate_data,
                                tolerance_m=float(config.self_collision_penetration_tolerance_m),
                            )
                        )
                        and (
                            not config.avoid_robot_object_collision
                            or not _has_robot_object_penetration(
                                model,
                                candidate_data,
                                tolerance_m=float(config.object_collision_penetration_tolerance_m),
                            )
                        )
                    )
                robot_ik.solve(model, data, ee_target, collision_ok=collision_ok)
                mujoco.mj_forward(model, data)
                self._last_robot_trace.append(
                    {
                        "time_s": float(time_s),
                        "target_world": [float(value) for value in ee_target],
                        "ee_world": [float(value) for value in data.xpos[robot_ik.ee_body_id]],
                        "ik_error": float(np.linalg.norm(np.asarray(data.xpos[robot_ik.ee_body_id], dtype=float) - ee_target)),
                        "qpos": [float(value) for value in data.qpos[robot_ik.qpos_adrs]],
                        "robot_object_contacts": _robot_object_contacts(model, data),
                        "robot_self_contacts": _robot_self_contacts(model, data),
                    }
                )
            self._update_scene(renderer, model, data, config)
            image = renderer.render()
            frames.append(
                self._overlay(
                    image,
                    episode_time=float(time_s),
                    q=float(q),
                    qdot=float(qdot),
                    target_q=target_q,
                    pulse_active=bool(pulse_active),
                    action_kind=action_kind,
                    frame_index=index,
                    frame_count=len(states),
                )
            )
        renderer.close()
        return frames

    def _dataset_states(
        self, response: np.ndarray, frame_times: np.ndarray, action_row: np.ndarray | None = None
    ) -> list[tuple[float, float, float, bool]]:
        src_t = np.asarray(response[:, 0], dtype=float)
        src_q = np.asarray(response[:, 1], dtype=float)
        src_qdot = np.asarray(response[:, 2], dtype=float)
        src_t = np.maximum.accumulate(src_t)
        if src_t[0] > 0.0:
            src_t = np.concatenate([[0.0], src_t])
            src_q = np.concatenate([[src_q[0]], src_q])
            src_qdot = np.concatenate([[src_qdot[0]], src_qdot])
        q = np.interp(frame_times, src_t, src_q)
        qdot = np.interp(frame_times, src_t, src_qdot)
        contact_duration = max(0.0, float(action_row[7])) if action_row is not None and action_row.shape[0] >= 8 else 0.0
        return [
            (float(t), float(a), float(b), bool(contact_duration > 0.0 and float(t) <= contact_duration + 1e-12))
            for t, a, b in zip(frame_times, q, qdot)
        ]

    def _resimulated_states(
        self,
        model,
        joint,
        q0: float,
        action_row: np.ndarray,
        frame_times: np.ndarray,
        object_visual_scale: float = 1.0,
    ) -> list[tuple[float, float, float, bool]]:
        data = mujoco.MjData(model)
        data.qpos[joint.qpos_adr] = float(np.clip(q0, joint.lower, joint.upper))
        data.qvel[joint.dof_adr] = float(action_row[0]) if action_row.shape[0] < 8 else 0.0
        mujoco.mj_forward(model, data)
        states: list[tuple[float, float, float, bool]] = []
        qfrc = np.zeros(model.nv, dtype=float)
        zero_torque = np.zeros(3, dtype=float)
        point = (
            np.asarray(action_row[:3], dtype=float) * max(1e-6, float(object_visual_scale))
            if action_row.shape[0] >= 8
            else np.zeros(3, dtype=float)
        )
        direction = _normalize(np.asarray(action_row[3:6], dtype=float)) if action_row.shape[0] >= 8 else np.zeros(3, dtype=float)
        force_mag = float(action_row[6]) if action_row.shape[0] >= 8 else 0.0
        contact_duration = max(0.0, float(action_row[7])) if action_row.shape[0] >= 8 else 0.0
        for target_time in frame_times:
            while data.time + 1e-12 < float(target_time):
                data.qfrc_applied[:] = 0.0
                if action_row.shape[0] >= 8 and data.time <= contact_duration + 1e-12 and abs(force_mag) > 0.0:
                    qfrc[:] = 0.0
                    mujoco.mj_applyFT(model, data, direction * force_mag, zero_torque, point, joint.body_id, qfrc)
                    data.qfrc_applied[:] = qfrc
                mujoco.mj_step(model, data)
            states.append(
                (
                    float(data.time),
                    float(data.qpos[joint.qpos_adr]),
                    float(data.qvel[joint.dof_adr]),
                    bool(action_row.shape[0] >= 8 and data.time <= contact_duration + 1e-12),
                )
            )
        return states

    def _update_scene(self, renderer, model, data, config: ILRolloutRenderConfig) -> None:
        if config.camera_name:
            renderer.update_scene(data, camera=str(config.camera_name))
            return
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.azimuth = float(config.camera_azimuth)
        camera.elevation = float(config.camera_elevation)
        camera.distance = float(config.camera_distance) if config.camera_distance is not None else max(0.5, float(model.stat.extent) * 2.5)
        camera.lookat[:] = np.asarray(model.stat.center, dtype=float)
        renderer.update_scene(data, camera=camera)

    def _overlay(
        self,
        image: np.ndarray,
        *,
        episode_time: float,
        q: float,
        qdot: float,
        target_q: float,
        pulse_active: bool,
        action_kind: str,
        frame_index: int,
        frame_count: int,
    ) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:
            return image
        pil = Image.fromarray(image)
        draw = ImageDraw.Draw(pil, "RGBA")
        font = ImageFont.load_default()
        lines = [
            f"t={episode_time:.2f}s  q={q:.3f}  target={target_q:.3f}  qdot={qdot:.3f}",
            f"{action_kind} | {'PULSE' if pulse_active else 'RELEASE'} | frame {frame_index + 1}/{frame_count}",
        ]
        pad = 10
        box_w = max(draw.textlength(line, font=font) for line in lines) + 2 * pad
        box_h = 42
        draw.rectangle((12, 12, 12 + box_w, 12 + box_h), fill=(255, 255, 255, 210))
        for i, line in enumerate(lines):
            draw.text((12 + pad, 18 + i * 16), line, font=font, fill=(17, 24, 39, 255))
        if pulse_active:
            draw.rectangle((pil.width - 112, 16, pil.width - 18, 42), fill=(220, 38, 38, 220))
            draw.text((pil.width - 96, 23), "PULSE", font=font, fill=(255, 255, 255, 255))
        return np.asarray(pil)

    def _write_video(self, output_path: Path, frames: list[np.ndarray], fps: int) -> None:
        import imageio.v2 as imageio

        suffix = output_path.suffix.lower()
        if suffix not in {".mp4", ".gif"}:
            output_path = output_path.with_suffix(".mp4")
        imageio.mimsave(output_path, frames, fps=max(1, int(fps)))

    def _write_frame_sequence(self, frame_dir: Path, frames: list[np.ndarray]) -> None:
        import imageio.v2 as imageio

        frame_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            imageio.imwrite(frame_dir / f"frame_{index:04d}.png", frame)

    def _write_ik_debug(self, output_path: Path, config: ILRolloutRenderConfig) -> None:
        trace = getattr(self, "_last_robot_trace", [])
        if not trace:
            return
        qpos = np.asarray([row["qpos"] for row in trace], dtype=float)
        errors = np.asarray([float(row["ik_error"]) for row in trace], dtype=float)
        object_contact_counts = np.asarray([len(row.get("robot_object_contacts", [])) for row in trace], dtype=float)
        object_penetrations = [
            max([-float(contact["dist"]) for contact in row.get("robot_object_contacts", []) if float(contact["dist"]) < 0.0], default=0.0)
            for row in trace
        ]
        self_contact_counts = np.asarray([len(row.get("robot_self_contacts", [])) for row in trace], dtype=float)
        self_penetrations = [
            max([-float(contact["dist"]) for contact in row.get("robot_self_contacts", []) if float(contact["dist"]) < 0.0], default=0.0)
            for row in trace
        ]
        save_json(
            {
                "source": "render-il-rollout-ik-debug",
                "robot_motion": config.robot_motion,
                "robot_ee_body": config.robot_ee_body,
                "num_frames": len(trace),
                "max_abs_joint_delta": float(np.max(np.abs(qpos - qpos[0]))),
                "max_frame_joint_delta": float(np.max(np.abs(np.diff(qpos, axis=0)))) if len(qpos) > 1 else 0.0,
                "ik_error_mean": float(np.mean(errors)),
                "ik_error_max": float(np.max(errors)),
                "robot_object_contact_count_max": int(np.max(object_contact_counts)) if len(object_contact_counts) else 0,
                "robot_object_penetration_max_m": float(np.max(object_penetrations)) if object_penetrations else 0.0,
                "robot_self_contact_count_max": int(np.max(self_contact_counts)) if len(self_contact_counts) else 0,
                "robot_self_penetration_max_m": float(np.max(self_penetrations)) if self_penetrations else 0.0,
                "frames": trace,
            },
            output_path,
        )

    def _output_path(self, dataset_path: Path, config: ILRolloutRenderConfig) -> Path:
        if config.output_path is not None:
            return Path(config.output_path).expanduser().resolve()
        return dataset_path.with_name(f"{dataset_path.stem}_episode_{int(config.episode_index):04d}.mp4")


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    return vec / norm


def _robot_object_contacts(model, data) -> list[dict[str, object]]:
    contacts: list[dict[str, object]] = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1])) or ""
        body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2])) or ""
        robot_body = body1 if body1.startswith("x5_") else body2 if body2.startswith("x5_") else ""
        object_body = body1 if body1.startswith("Microwave") else body2 if body2.startswith("Microwave") else ""
        if not robot_body or not object_body:
            continue
        contacts.append(
            {
                "geom1": geom1,
                "geom2": geom2,
                "body1": body1,
                "body2": body2,
                "robot_body": robot_body,
                "object_body": object_body,
                "dist": float(contact.dist),
            }
        )
    return contacts


def _has_robot_object_penetration(model, data, tolerance_m: float = 0.0) -> bool:
    tolerance = max(0.0, float(tolerance_m))
    return any(float(contact["dist"]) < -tolerance for contact in _robot_object_contacts(model, data))


def _robot_self_contacts(model, data) -> list[dict[str, object]]:
    contacts: list[dict[str, object]] = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1_id = int(model.geom_bodyid[geom1])
        body2_id = int(model.geom_bodyid[geom2])
        body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id) or ""
        body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id) or ""
        if not body1.startswith("x5_") or not body2.startswith("x5_"):
            continue
        if _is_ignored_robot_self_pair(model, body1_id, body2_id, body1, body2):
            continue
        contacts.append(
            {
                "geom1": geom1,
                "geom2": geom2,
                "body1": body1,
                "body2": body2,
                "dist": float(contact.dist),
            }
        )
    return contacts


def _has_robot_self_penetration(model, data, tolerance_m: float = 0.0) -> bool:
    tolerance = max(0.0, float(tolerance_m))
    return any(float(contact["dist"]) < -tolerance for contact in _robot_self_contacts(model, data))


def _is_direct_parent_child(model, body1_id: int, body2_id: int) -> bool:
    return int(model.body_parentid[body1_id]) == int(body2_id) or int(model.body_parentid[body2_id]) == int(body1_id)


def _is_ignored_robot_self_pair(model, body1_id: int, body2_id: int, body1: str, body2: str) -> bool:
    if _is_direct_parent_child(model, body1_id, body2_id):
        return True
    # X5 link7/link8 are sibling gripper/finger links. Their exported STL
    # collision meshes overlap at the default opening, so treating this pair as
    # self-collision blocks every IK solve without improving arm placement.
    return {body1, body2} == {"x5_link7", "x5_link8"}


@dataclass(slots=True)
class _RobotIKState:
    ee_body_id: int
    joint_ids: list[int]
    qpos_adrs: np.ndarray
    dof_adrs: np.ndarray

    @classmethod
    def create(cls, model, ee_body_name: str) -> "_RobotIKState":
        candidates = [ee_body_name]
        if not ee_body_name.startswith("x5_"):
            candidates.append("x5_" + ee_body_name)
        ee_body_id = -1
        for name in candidates:
            ee_body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
            if ee_body_id >= 0:
                break
        if ee_body_id < 0:
            raise ValueError(f"Could not find robot end-effector body: {ee_body_name!r}")
        joint_ids = []
        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
            if name.startswith("x5_joint"):
                joint_type = int(model.jnt_type[jid])
                if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
                    joint_ids.append(jid)
        if not joint_ids:
            raise ValueError("Combined scene does not contain prefixed X5 joints")
        return cls(
            ee_body_id=ee_body_id,
            joint_ids=joint_ids,
            qpos_adrs=np.asarray([int(model.jnt_qposadr[jid]) for jid in joint_ids], dtype=int),
            dof_adrs=np.asarray([int(model.jnt_dofadr[jid]) for jid in joint_ids], dtype=int),
        )

    def solve(self, model, data, target_world: np.ndarray, collision_ok: Callable[[object], bool] | None = None) -> None:
        target = np.asarray(target_world, dtype=float)
        damping = 1e-4
        best_q = np.asarray(data.qpos[self.qpos_adrs], dtype=float).copy()
        mujoco.mj_forward(model, data)
        best_err = float(np.linalg.norm(target - np.asarray(data.xpos[self.ee_body_id], dtype=float)))
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        for _ in range(80):
            mujoco.mj_forward(model, data)
            ee = np.asarray(data.xpos[self.ee_body_id], dtype=float)
            err = target - ee
            current_err = float(np.linalg.norm(err))
            if current_err < 1e-3:
                return
            jacp[:] = 0.0
            jacr[:] = 0.0
            mujoco.mj_jacBody(model, data, jacp, jacr, self.ee_body_id)
            jac = jacp[:, self.dof_adrs]
            if float(np.linalg.norm(jac)) < 1e-9:
                break
            jj_t = jac @ jac.T
            step = jac.T @ np.linalg.solve(jj_t + damping * np.eye(3), err)
            step = np.clip(step, -0.12, 0.12)
            base_q = np.asarray(data.qpos[self.qpos_adrs], dtype=float).copy()
            accepted = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate_q = self._clip_joint_positions(model, base_q + scale * step)
                data.qpos[self.qpos_adrs] = candidate_q
                mujoco.mj_forward(model, data)
                candidate_err = float(np.linalg.norm(target - np.asarray(data.xpos[self.ee_body_id], dtype=float)))
                if candidate_err + 1e-6 < current_err and (collision_ok is None or collision_ok(data)):
                    accepted = True
                    if candidate_err < best_err:
                        best_err = candidate_err
                        best_q = candidate_q.copy()
                    break
            if not accepted:
                data.qpos[self.qpos_adrs] = best_q
                break
        data.qpos[self.qpos_adrs] = best_q

    def _clip_joint_positions(self, model, values: np.ndarray) -> np.ndarray:
        clipped = np.asarray(values, dtype=float).copy()
        for index, jid in enumerate(self.joint_ids):
            if bool(model.jnt_limited[jid]):
                clipped[index] = float(np.clip(clipped[index], float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])))
        return clipped


def _robot_pulse_target(
    action_row: np.ndarray,
    time_s: float,
    approach_distance: float,
    pulse_distance: float,
    point_scale: float = 1.0,
) -> np.ndarray:
    point = np.asarray(action_row[:3], dtype=float) * max(1e-6, float(point_scale))
    direction = _normalize(np.asarray(action_row[3:6], dtype=float))
    contact_duration = max(1e-6, float(action_row[7]))
    if time_s <= contact_duration:
        alpha = float(np.clip(time_s / contact_duration, 0.0, 1.0))
        return point + direction * (alpha * max(0.0, pulse_distance))
    release_alpha = float(np.clip((time_s - contact_duration) / max(0.2, 2.0 * contact_duration), 0.0, 1.0))
    return point + direction * (max(0.0, pulse_distance) - release_alpha * max(0.0, approach_distance))


def _write_combined_object_robot_mjcf(
    *,
    object_mjcf: Path,
    robot_model: Path,
    output_path: Path,
    robot_base_pos: tuple[float, float, float],
    robot_base_euler: tuple[float, float, float],
    object_visual_scale: float = 1.0,
) -> None:
    object_tree = ET.parse(object_mjcf)
    object_root = object_tree.getroot()
    compiler = object_root.find("compiler")
    if compiler is not None:
        _absolutize_compiler_path(compiler, "assetdir", object_mjcf.parent)
        _absolutize_compiler_path(compiler, "texturedir", object_mjcf.parent)
    if abs(float(object_visual_scale) - 1.0) > 1e-9:
        _scale_object_mjcf(object_root, float(object_visual_scale))

    robot_root = _robot_as_mjcf_root(robot_model)
    _prefix_robot_mjcf(robot_root, "x5_")
    _absolutize_robot_mesh_files(robot_root, robot_model.parent)

    object_asset = object_root.find("asset")
    robot_asset = robot_root.find("asset")
    if robot_asset is not None:
        if object_asset is None:
            object_asset = ET.SubElement(object_root, "asset")
        for child in list(robot_asset):
            object_asset.append(child)

    object_worldbody = object_root.find("worldbody")
    robot_worldbody = robot_root.find("worldbody")
    if object_worldbody is None or robot_worldbody is None:
        raise ValueError("Both object and robot MJCFs must contain worldbody")
    wrapper = ET.Element(
        "body",
        {
            "name": "x5_base_visual",
            "pos": _fmt_vec(robot_base_pos),
            "euler": _fmt_vec(robot_base_euler),
        },
    )
    for child in list(robot_worldbody):
        wrapper.append(child)
    object_worldbody.append(wrapper)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(object_tree, space="  ")
    object_tree.write(output_path, encoding="utf-8", xml_declaration=False)


def _robot_as_mjcf_root(robot_model: Path) -> ET.Element:
    suffix = robot_model.suffix.lower()
    if suffix == ".urdf":
        model = mujoco.MjModel.from_xml_path(str(robot_model))
        with tempfile.TemporaryDirectory() as temp_dir:
            converted = Path(temp_dir) / "robot_converted.xml"
            mujoco.mj_saveLastXML(str(converted), model)
            return ET.fromstring(converted.read_text(encoding="utf-8"))
    return ET.parse(robot_model).getroot()


def _prefix_robot_mjcf(root: ET.Element, prefix: str) -> None:
    named_tags = {"body", "joint", "geom", "site", "camera", "light", "mesh", "material", "texture"}
    ref_attrs = {"mesh", "material", "texture"}
    for elem in root.iter():
        if elem.tag in named_tags and elem.get("name"):
            elem.set("name", prefix + str(elem.get("name")))
        for attr in ref_attrs:
            if elem.get(attr):
                elem.set(attr, prefix + str(elem.get(attr)))


def _scale_object_mjcf(root: ET.Element, scale: float) -> None:
    scale = max(1e-6, float(scale))
    for mesh in root.findall(".//asset/mesh"):
        existing = _parse_vec(mesh.get("scale"), default=(1.0, 1.0, 1.0))
        mesh.set("scale", _fmt_vec(tuple(scale * value for value in existing)))

    worldbody = root.find("worldbody")
    if worldbody is None:
        return
    spatial_tags = {"body", "geom", "joint", "site", "camera", "light", "inertial"}
    for child in list(worldbody):
        if child.tag != "body":
            continue
        for elem in child.iter():
            if elem.tag in spatial_tags and elem.get("pos"):
                values = _parse_vec(elem.get("pos"), default=(0.0, 0.0, 0.0))
                elem.set("pos", _fmt_vec(tuple(scale * value for value in values)))
            if elem.tag == "inertial" and elem.get("diaginertia"):
                inertia = _parse_vec(elem.get("diaginertia"), default=(1.0, 1.0, 1.0))
                elem.set("diaginertia", _fmt_vec(tuple((scale**2) * value for value in inertia)))


def _absolutize_robot_mesh_files(root: ET.Element, base_dir: Path) -> None:
    for mesh in root.findall(".//mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue
        path = Path(file_attr)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        mesh.set("file", str(path))


def _absolutize_compiler_path(compiler: ET.Element, attr: str, base_dir: Path) -> None:
    value = compiler.get(attr)
    if not value:
        return
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    compiler.set(attr, str(path))


def _fmt_vec(values: tuple[float, float, float]) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _parse_vec(text: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not text:
        return default
    parts = [float(part) for part in str(text).split()]
    if len(parts) < 3:
        parts = [*parts, *list(default[len(parts) :])]
    return float(parts[0]), float(parts[1]), float(parts[2])
