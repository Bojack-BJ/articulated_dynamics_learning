from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.categories import FULL_ROTATION_CATEGORIES, HINGE_CATEGORIES, PRISMATIC_CATEGORIES, SUPPORTED_CATEGORIES
from ..core.serialization import load_json, save_json
from ..perception.part_segmentation import build_mujoco_body_part_segmentation, segmentation_to_part_mask_u16


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vec))
    if norm < 1e-9:
        return [0.0, 0.0, 0.0]
    return [component / norm for component in vec]


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def orbit_camera_pose(
    lookat: list[float],
    distance: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> list[list[float]]:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    radius_xy = distance * math.cos(elevation)

    position = [
        lookat[0] - radius_xy * math.cos(azimuth),
        lookat[1] - radius_xy * math.sin(azimuth),
        lookat[2] - distance * math.sin(elevation),
    ]

    forward = _normalize([lookat[index] - position[index] for index in range(3)])
    world_up = [0.0, 0.0, 1.0]
    right = _normalize(_cross(forward, world_up))
    if sum(component * component for component in right) < 1e-8:
        world_up = [0.0, 1.0, 0.0]
        right = _normalize(_cross(forward, world_up))
    up = _cross(right, forward)

    return [
        [right[0], up[0], forward[0], position[0]],
        [right[1], up[1], forward[1], position[1]],
        [right[2], up[2], forward[2], position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _write_ppm(path: Path, rgb: Any) -> None:
    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {channels}")
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + rgb.tobytes())


def _write_pgm_u16(path: Path, depth_u16: Any) -> None:
    height, width = depth_u16.shape
    header = f"P5\n{width} {height}\n65535\n".encode("ascii")
    path.write_bytes(header + depth_u16.astype(">u2", copy=False).tobytes())


def _write_png_rgb(path: Path, rgb: Any) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PNG writing requires Pillow. Install with: pip install -e '.[viz]'") from exc

    image = Image.fromarray(rgb, mode="RGB")
    image.save(path)


def _write_png_depth_u16(path: Path, depth_u16: Any) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PNG writing requires Pillow. Install with: pip install -e '.[viz]'") from exc

    image = Image.fromarray(depth_u16, mode="I;16")
    image.save(path)


def _read_ppm_rgb(path: Path):
    import numpy as np

    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P6":
            raise ValueError(f"Unsupported PPM magic for {path}: {magic!r}")

        tokens: list[bytes] = []
        while len(tokens) < 3:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            tokens.extend(line.split())
        if len(tokens) < 3:
            raise ValueError(f"Incomplete PPM header in {path}")

        width = int(tokens[0])
        height = int(tokens[1])
        max_value = int(tokens[2])
        if max_value != 255:
            raise ValueError(f"Unsupported PPM max value in {path}: {max_value}")

        raw = handle.read(width * height * 3)
        if len(raw) != width * height * 3:
            raise ValueError(f"Incomplete PPM payload in {path}")
        return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))


def _read_pgm_u16(path: Path):
    import numpy as np

    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Unsupported PGM magic for {path}: {magic!r}")

        tokens: list[bytes] = []
        while len(tokens) < 3:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            tokens.extend(line.split())
        if len(tokens) < 3:
            raise ValueError(f"Incomplete PGM header in {path}")

        width = int(tokens[0])
        height = int(tokens[1])
        max_value = int(tokens[2])
        if max_value > 65535:
            raise ValueError(f"Unsupported PGM max value in {path}: {max_value}")

        if max_value <= 255:
            raw = handle.read(width * height)
            if len(raw) != width * height:
                raise ValueError(f"Incomplete 8-bit PGM payload in {path}")
            return np.frombuffer(raw, dtype=np.uint8).astype(np.uint16).reshape((height, width))

        raw = handle.read(width * height * 2)
        if len(raw) != width * height * 2:
            raise ValueError(f"Incomplete 16-bit PGM payload in {path}")
        return np.frombuffer(raw, dtype=">u2").astype(np.uint16).reshape((height, width))


def _read_png_rgb(path: Path):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PNG RGB loading requires Pillow.") from exc

    import numpy as np

    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _read_png_u16(path: Path):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PNG depth loading requires Pillow.") from exc

    import numpy as np

    return np.asarray(Image.open(path), dtype=np.uint16)


def _intrinsics_from_fovy(width: int, height: int, fovy_deg: float) -> dict[str, float]:
    fovy_rad = math.radians(fovy_deg)
    fy = 0.5 * float(height) / math.tan(0.5 * fovy_rad)
    fx = fy
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def _fovy_deg_from_intrinsics(height: int, camera_intrinsics: dict[str, Any]) -> float | None:
    fy = camera_intrinsics.get("fy")
    if fy is None:
        return None
    fy_value = float(fy)
    if fy_value <= 0.0 or height <= 0:
        return None
    return math.degrees(2.0 * math.atan(0.5 * float(height) / fy_value))


@dataclass(slots=True)
class MuJoCoRecordConfig:
    model_path: str | Path
    output_dir: str | Path
    object_instance_id: str
    category: str
    frame_count: int = 48
    sim_dt: float = 1.0 / 240.0
    frame_dt: float = 0.1
    width: int = 640
    height: int = 480
    rgb_format: str = "ppm"
    depth_format: str = "pgm"
    camera_distance: float = 1.8
    camera_elevation_deg: float = 22.0
    camera_azimuth_start_deg: float = 20.0
    camera_azimuth_span_deg: float = 140.0
    camera_fovy_deg: float = 45.0
    camera_mode: str = "orbit"  # 'orbit' or 'triview'
    camera_triview_spacing_deg: float = 45.0
    lookat: tuple[float, float, float] = (0.0, 0.0, 0.8)
    perturbation_scale: float = 0.3
    control_kp: float = 30.0
    control_kd: float = 3.0
    seed: int = 0
    control_mode: str = "track"  # 'track' or 'free'
    random_initial_qpos: bool = False
    auto_initial_qvel_from_limits: bool = False
    auto_initial_qvel_direction_mode: str = "away-from-qpos0"
    auto_initial_qvel_min_abs: float = 1.0
    auto_initial_qvel_max_abs: float = 8.0
    initial_joint_qvel: float = 0.0
    staged_initial_qvel: bool = False
    staged_primary_tokens: tuple[str, ...] = ("door",)
    staged_secondary_tokens: tuple[str, ...] = ("drawer", "slide")
    staged_secondary_count_min: int = 1
    staged_secondary_count_max: int = 2
    staged_secondary_start_s: float = 1.2
    kick_force: float = 0.0
    kick_start_s: float = 0.0
    kick_duration_s: float = 0.0
    excitation_mode: str = "none"
    excitation_force: float = 0.0
    excitation_start_s: float = 0.0
    excitation_duration_s: float = 0.0
    excitation_period_s: float = 0.25
    excitation_frequency_hz: float = 1.0
    write_dynamics_log: bool = True
    make_video: bool = False
    video_fps: float | None = None
    segmentation_masks: bool = False
    part_segmentation_masks: bool = False
    mask_format: str = "pgm"
    write_concat_assets: bool = False
    disable_target_mesh_collision: bool = False
    hide_clear_meshes: bool = False
    joint_name: str | None = None
    joint_id: int | None = None
    all_joints: bool = False


@dataclass(slots=True)
class MuJoCoMaskRenderConfig:
    episode_path: str | Path
    model_path: str | Path | None = None
    mask_format: str = "pgm"
    part_segmentation_masks: bool = False
    output_episode_path: str | Path | None = None


@dataclass(slots=True)
class MuJoCoEpisodeCompactConfig:
    episode_path: str | Path
    remove_concat_dir: bool = True
    remove_concat_video: bool = False
    dry_run: bool = False


@dataclass(slots=True)
class MuJoCoEpisodeRepackConfig:
    episode_path: str | Path
    keep_originals: bool = False
    dry_run: bool = False


def _triview_writes_concat_assets(camera_mode: str, write_concat_assets: bool) -> bool:
    return camera_mode != "triview" or bool(write_concat_assets)


def _primary_view_path(paths: list[Path] | list[str], fallback: Path | str | None) -> Path | str | None:
    if paths:
        return paths[len(paths) // 2]
    return fallback


class MuJoCoEpisodeRecorder:
    def __init__(self, config: MuJoCoRecordConfig) -> None:
        self.config = config

    def record(self) -> Path:
        if self.config.category not in SUPPORTED_CATEGORIES:
            raise ValueError(f"category must be one of: {', '.join(SUPPORTED_CATEGORIES)}")
        if self.config.frame_count < 1:
            raise ValueError("frame_count must be >= 1")
        if self.config.sim_dt <= 0.0 or self.config.frame_dt <= 0.0:
            raise ValueError("sim_dt and frame_dt must be positive")
        if self.config.rgb_format not in {"ppm", "png"}:
            raise ValueError("rgb_format must be 'ppm' or 'png'")
        if self.config.depth_format not in {"pgm", "png"}:
            raise ValueError("depth_format must be 'pgm' or 'png'")
        if self.config.control_mode not in {"track", "free"}:
            raise ValueError("control_mode must be 'track' or 'free'")
        if self.config.camera_mode not in {"orbit", "triview"}:
            raise ValueError("camera_mode must be 'orbit' or 'triview'")
        if self.config.camera_triview_spacing_deg <= 0.0:
            raise ValueError("camera_triview_spacing_deg must be positive")
        if self.config.kick_duration_s < 0.0:
            raise ValueError("kick_duration_s must be >= 0")
        if self.config.kick_start_s < 0.0:
            raise ValueError("kick_start_s must be >= 0")
        if self.config.excitation_mode not in {"none", "pulse", "prbs", "sine"}:
            raise ValueError("excitation_mode must be one of: none, pulse, prbs, sine")
        if self.config.excitation_start_s < 0.0:
            raise ValueError("excitation_start_s must be >= 0")
        if self.config.excitation_duration_s < 0.0:
            raise ValueError("excitation_duration_s must be >= 0")
        if self.config.excitation_mode != "none" and self.config.excitation_duration_s <= 0.0:
            raise ValueError("excitation_duration_s must be > 0 when excitation_mode is not 'none'")
        if self.config.excitation_period_s <= 0.0:
            raise ValueError("excitation_period_s must be positive")
        if self.config.excitation_frequency_hz <= 0.0:
            raise ValueError("excitation_frequency_hz must be positive")
        if self.config.auto_initial_qvel_direction_mode not in {
            "away-from-qpos0",
            "toward-lower",
            "toward-upper",
        }:
            raise ValueError(
                "auto_initial_qvel_direction_mode must be one of: away-from-qpos0, toward-lower, toward-upper"
            )
        if self.config.auto_initial_qvel_min_abs < 0.0:
            raise ValueError("auto_initial_qvel_min_abs must be >= 0")
        if self.config.auto_initial_qvel_max_abs <= 0.0:
            raise ValueError("auto_initial_qvel_max_abs must be > 0")
        if self.config.video_fps is not None and self.config.video_fps <= 0.0:
            raise ValueError("video_fps must be positive")
        if self.config.mask_format not in {"pgm", "png"}:
            raise ValueError("mask_format must be 'pgm' or 'png'")
        if self.config.all_joints and self.config.control_mode != "free":
            raise ValueError("all_joints requires control_mode='free'")
        if self.config.staged_initial_qvel and self.config.control_mode != "free":
            raise ValueError("staged_initial_qvel requires control_mode='free'")
        if self.config.staged_initial_qvel and (self.config.joint_name is not None or self.config.joint_id is not None):
            raise ValueError("staged_initial_qvel cannot be combined with joint_name/joint_id")
        if self.config.all_joints and (self.config.joint_name is not None or self.config.joint_id is not None):
            raise ValueError("all_joints cannot be combined with joint_name/joint_id")
        if self.config.joint_name is not None and self.config.joint_id is not None:
            raise ValueError("Provide only one of joint_name or joint_id")

        mujoco = self._import_mujoco()
        import numpy as np

        model = mujoco.MjModel.from_xml_path(str(Path(self.config.model_path)))
        data = mujoco.MjData(model)
        model.opt.timestep = self.config.sim_dt

        model.vis.global_.fovy = self.config.camera_fovy_deg
        scene_option = None
        renderer = mujoco.Renderer(model, width=self.config.width, height=self.config.height)

        controlled = self._select_controlled_joints(mujoco, model)
        primary_joint_id, primary_qpos_adr, primary_dof_adr = controlled[0]
        target_root_body_id = self._infer_object_root_body_id(model, controlled)
        target_geom_ids = self._collect_target_geom_ids(model, target_root_body_id)
        if self.config.disable_target_mesh_collision:
            self._disable_target_mesh_collision(model, target_geom_ids)
        hidden_clear_geom_ids: set[int] = set()
        if self.config.hide_clear_meshes:
            hidden_clear_geom_ids = self._collect_clear_mesh_geom_ids(mujoco, model, target_geom_ids)
            scene_option = self._scene_option_hiding_geom_ids(mujoco, model, hidden_clear_geom_ids)
        part_segmentation = (
            build_mujoco_body_part_segmentation(
                mujoco=mujoco,
                model=model,
                root_body_id=target_root_body_id,
                target_geom_ids=target_geom_ids,
                hidden_geom_ids=hidden_clear_geom_ids,
            )
            if self.config.part_segmentation_masks
            else None
        )
        joint_limits = self._joint_limits(model, primary_joint_id)
        rng = random.Random(self.config.seed)

        record_duration_s = max(self.config.frame_count * self.config.frame_dt, self.config.frame_dt)

        for index, (joint_id, qpos_adr, dof_adr) in enumerate(controlled):
            if self.config.control_mode == "free":
                self._initialize_free_joint_state(
                    mujoco,
                    model,
                    data,
                    joint_id,
                    qpos_adr,
                    dof_adr,
                    primary_joint_id,
                    index,
                    rng,
                )
            else:
                data.qvel[dof_adr] = float(self.config.initial_joint_qvel)

        if self.config.control_mode == "free":
            mujoco.mj_forward(model, data)

        output_dir = Path(self.config.output_dir).resolve() / self.config.object_instance_id
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        write_concat_assets = _triview_writes_concat_assets(
            self.config.camera_mode,
            self.config.write_concat_assets,
        )
        concat_assets_dir = assets_dir
        per_view_assets_dirs: list[Path] = []
        if self.config.camera_mode == "triview":
            per_view_assets_dirs = [assets_dir / f"view_{index}" for index in range(3)]
            for view_dir in per_view_assets_dirs:
                view_dir.mkdir(parents=True, exist_ok=True)
            if write_concat_assets:
                concat_assets_dir = assets_dir / "concat"
                concat_assets_dir.mkdir(parents=True, exist_ok=True)

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE

        steps_per_frame = max(1, int(round(self.config.frame_dt / self.config.sim_dt)))

        triview_azimuths = [
            self.config.camera_azimuth_start_deg - self.config.camera_triview_spacing_deg,
            self.config.camera_azimuth_start_deg,
            self.config.camera_azimuth_start_deg + self.config.camera_triview_spacing_deg,
        ]

        frame_payload: list[dict[str, Any]] = []
        dynamics_log_rows: list[dict[str, Any]] = []
        concat_video_writer = None
        view_video_writers: list[Any] = []
        if self.config.make_video:
            fps = self.config.video_fps if self.config.video_fps is not None else 1.0 / self.config.frame_dt
            if self.config.camera_mode == "triview":
                concat_video_writer = self._open_video_writer(
                    output_dir=output_dir,
                    fps=fps,
                    filename="episode_concat.mp4",
                )
                view_video_writers = [
                    self._open_video_writer(
                        output_dir=output_dir,
                        fps=fps,
                        filename=f"episode_view_{index}.mp4",
                    )
                    for index in range(3)
                ]
            else:
                concat_video_writer = self._open_video_writer(
                    output_dir=output_dir,
                    fps=fps,
                    filename="episode.mp4",
                )
        previous_target = joint_limits[0]
        previous_tracking_force = 0.0
        joint_names = [self._joint_name(mujoco, model, joint_id) for joint_id, _, _ in controlled]
        staged_secondary_joints = self._select_staged_secondary_joints(mujoco, model, controlled, rng)
        staged_secondary_triggered = False

        try:
            for frame_index in range(self.config.frame_count):
                alpha = 0.0 if self.config.frame_count == 1 else frame_index / (self.config.frame_count - 1)
                target_fraction = 0.5 - 0.5 * math.cos(alpha * math.pi)
                target_q = joint_limits[0] + target_fraction * (joint_limits[1] - joint_limits[0])

                force_history: list[float] = []
                excitation_force_history: list[float] = []
                for _ in range(steps_per_frame):
                    step_time_s = float(data.time)
                    step_forces: list[float] = []
                    step_excitation_forces: list[float] = []
                    data.qfrc_applied[:] = 0.0

                    if self.config.staged_initial_qvel and not staged_secondary_triggered:
                        if step_time_s >= float(self.config.staged_secondary_start_s):
                            for joint_index, (joint_id, qpos_adr, dof_adr) in enumerate(controlled):
                                if joint_id not in staged_secondary_joints:
                                    continue
                                lower, upper = self._joint_limits(model, joint_id)
                                if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
                                    desired_speed = 0.25 * (upper - lower) / self.config.frame_dt
                                else:
                                    desired_speed = abs(float(self.config.initial_joint_qvel))
                                data.qvel[dof_adr] = self._free_initial_qvel(
                                    model,
                                    data,
                                    joint_id,
                                    qpos_adr,
                                    joint_index,
                                    desired_speed,
                                    lower,
                                    upper,
                                )
                            staged_secondary_triggered = True

                    kick_force = 0.0
                    if self.config.control_mode != "free" and self.config.kick_duration_s > 0.0:
                        if self.config.kick_start_s <= step_time_s < (self.config.kick_start_s + self.config.kick_duration_s):
                            kick_force = float(self.config.kick_force)

                    joint_step_logs: dict[str, dict[str, float]] = {}
                    for joint_index, (joint_id, qpos_adr, dof_adr) in enumerate(controlled):
                        q = float(data.qpos[qpos_adr])
                        qdot = float(data.qvel[dof_adr])
                        joint_name = self._joint_name(mujoco, model, joint_id)

                        tracking_force = 0.0
                        if self.config.control_mode == "track" and joint_id == primary_joint_id:
                            # Use simulation time for a continuous trajectory instead of frame-wise steps.
                            if record_duration_s <= 0.0:
                                phase = 1.0
                            else:
                                phase = min(max(float(data.time) / record_duration_s, 0.0), 1.0)
                            continuous_fraction = 0.5 - 0.5 * math.cos(phase * math.pi)
                            target_q = joint_limits[0] + continuous_fraction * (joint_limits[1] - joint_limits[0])

                            raw_tracking_force = self.config.control_kp * (target_q - q) - self.config.control_kd * qdot
                            # A light low-pass filter suppresses high-frequency jitter in rendered motion.
                            tracking_force = 0.85 * previous_tracking_force + 0.15 * raw_tracking_force
                            previous_tracking_force = tracking_force

                        perturb_force = (
                            rng.uniform(-1.0, 1.0) * self.config.perturbation_scale
                            if self.config.control_mode != "free"
                            else 0.0
                        )
                        excitation_force = self._excitation_force(step_time_s, joint_index)
                        total_force = tracking_force + perturb_force + kick_force + excitation_force
                        data.qfrc_applied[dof_adr] = total_force
                        step_forces.append(total_force)
                        step_excitation_forces.append(excitation_force)
                        joint_step_logs[joint_name] = {
                            "qpos": q,
                            "qvel": qdot,
                            "tracking_force": tracking_force,
                            "perturbation_force": perturb_force,
                            "kick_force": kick_force,
                            "excitation_force": excitation_force,
                            "applied_generalized_force": total_force,
                        }

                    force_history.append(float(sum(step_forces) / len(step_forces)) if step_forces else 0.0)
                    excitation_force_history.append(
                        float(sum(step_excitation_forces) / len(step_excitation_forces))
                        if step_excitation_forces
                        else 0.0
                    )
                    if self.config.write_dynamics_log:
                        dynamics_log_rows.append(
                            {
                                "timestamp_s": round(step_time_s, 9),
                                "joints": joint_step_logs,
                            }
                        )
                    mujoco.mj_step(model, data)

                    if self.config.control_mode == "free":
                        for joint_id, qpos_adr, dof_adr in controlled:
                            lower, upper = self._joint_limits(model, joint_id)
                            if not (math.isfinite(lower) and math.isfinite(upper) and upper > lower):
                                continue
                            q = float(data.qpos[qpos_adr])
                            if q < lower:
                                data.qpos[qpos_adr] = lower
                                data.qvel[dof_adr] = 0.0
                            elif q > upper:
                                data.qpos[qpos_adr] = upper
                                data.qvel[dof_adr] = 0.0
                        mujoco.mj_forward(model, data)

                if self.config.camera_mode == "orbit":
                    azimuths = [self.config.camera_azimuth_start_deg + self.config.camera_azimuth_span_deg * alpha]
                else:
                    azimuths = triview_azimuths

                rgb_views = []
                depth_views_u16 = []
                view_camera_poses = []
                for azimuth in azimuths:
                    camera.azimuth = azimuth
                    camera.elevation = self.config.camera_elevation_deg
                    camera.distance = self.config.camera_distance
                    camera.lookat[:] = list(self.config.lookat)

                    renderer.update_scene(data, camera=camera, scene_option=scene_option)
                    view_camera_poses.append(self._camera_pose_from_scene_camera(renderer.scene.camera[0]))
                    rgb_views.append(renderer.render())

                    renderer.enable_depth_rendering()
                    renderer.update_scene(data, camera=camera, scene_option=scene_option)
                    depth = renderer.render()
                    renderer.disable_depth_rendering()
                    depth_views_u16.append(np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16))

                if len(rgb_views) == 1:
                    rgb = rgb_views[0]
                    depth_u16 = depth_views_u16[0]
                else:
                    rgb = np.concatenate(rgb_views, axis=1)
                    depth_u16 = np.concatenate(depth_views_u16, axis=1)

                rgb_ext = "ppm" if self.config.rgb_format == "ppm" else "png"
                depth_ext = "pgm" if self.config.depth_format == "pgm" else "png"
                mask_ext = "pgm" if self.config.mask_format == "pgm" else "png"
                rgb_name = f"frame_{frame_index:04d}_rgb.{rgb_ext}"
                depth_name = f"frame_{frame_index:04d}_depth.{depth_ext}"
                mask_name = f"frame_{frame_index:04d}_mask.{mask_ext}"

                rgb_path = None
                depth_path = None
                if write_concat_assets:
                    rgb_path = concat_assets_dir / rgb_name
                    depth_path = concat_assets_dir / depth_name
                    if self.config.rgb_format == "ppm":
                        _write_ppm(rgb_path, rgb)
                    else:
                        _write_png_rgb(rgb_path, rgb)

                    if self.config.depth_format == "pgm":
                        _write_pgm_u16(depth_path, depth_u16)
                    else:
                        _write_png_depth_u16(depth_path, depth_u16)

                view_rgb_paths: list[Path] = []
                view_depth_paths: list[Path] = []
                view_mask_paths: list[Path] = []
                view_part_mask_paths: list[Path] = []
                view_masks_u16: list[Any] = []
                view_part_masks_u16: list[Any] = []
                if self.config.camera_mode == "triview":
                    for view_index, (view_rgb, view_depth_u16) in enumerate(zip(rgb_views, depth_views_u16)):
                        view_rgb_path = per_view_assets_dirs[view_index] / rgb_name
                        view_depth_path = per_view_assets_dirs[view_index] / depth_name
                        if self.config.rgb_format == "ppm":
                            _write_ppm(view_rgb_path, view_rgb)
                        else:
                            _write_png_rgb(view_rgb_path, view_rgb)
                        if self.config.depth_format == "pgm":
                            _write_pgm_u16(view_depth_path, view_depth_u16)
                        else:
                            _write_png_depth_u16(view_depth_path, view_depth_u16)
                        view_rgb_paths.append(view_rgb_path)
                        view_depth_paths.append(view_depth_path)

                write_binary_masks = bool(self.config.segmentation_masks or self.config.part_segmentation_masks)
                part_mask_name = f"frame_{frame_index:04d}_part_mask.{mask_ext}"
                part_mask_path = None
                if write_binary_masks:
                    for azimuth in azimuths:
                        camera.azimuth = azimuth
                        camera.elevation = self.config.camera_elevation_deg
                        camera.distance = self.config.camera_distance
                        camera.lookat[:] = list(self.config.lookat)
                        renderer.enable_segmentation_rendering()
                        renderer.update_scene(data, camera=camera, scene_option=scene_option)
                        segmentation = renderer.render()
                        renderer.disable_segmentation_rendering()
                        if self.config.part_segmentation_masks and part_segmentation is not None:
                            view_part_mask_u16 = segmentation_to_part_mask_u16(
                                mujoco=mujoco,
                                segmentation=segmentation,
                                target_geom_ids=target_geom_ids,
                                part_segmentation=part_segmentation,
                            )
                            view_masks_u16.append(self._binary_mask_from_part_mask_u16(view_part_mask_u16))
                            view_part_masks_u16.append(view_part_mask_u16)
                        else:
                            view_masks_u16.append(
                                self._segmentation_to_binary_mask_u16(
                                    mujoco=mujoco,
                                    segmentation=segmentation,
                                    target_geom_ids=target_geom_ids,
                                )
                            )

                    if len(view_masks_u16) == 1:
                        mask_u16 = view_masks_u16[0]
                    else:
                        mask_u16 = np.concatenate(view_masks_u16, axis=1)

                    mask_path = None
                    if write_concat_assets:
                        mask_path = concat_assets_dir / mask_name
                        if self.config.mask_format == "pgm":
                            _write_pgm_u16(mask_path, mask_u16)
                        else:
                            _write_png_depth_u16(mask_path, mask_u16)

                    if self.config.part_segmentation_masks and view_part_masks_u16:
                        if len(view_part_masks_u16) == 1:
                            part_mask_u16 = view_part_masks_u16[0]
                        else:
                            part_mask_u16 = np.concatenate(view_part_masks_u16, axis=1)
                        if write_concat_assets:
                            part_mask_path = concat_assets_dir / part_mask_name
                            if self.config.mask_format == "pgm":
                                _write_pgm_u16(part_mask_path, part_mask_u16)
                            else:
                                _write_png_depth_u16(part_mask_path, part_mask_u16)

                    if self.config.camera_mode == "triview":
                        for view_index, view_mask_u16 in enumerate(view_masks_u16):
                            view_mask_path = per_view_assets_dirs[view_index] / mask_name
                            if self.config.mask_format == "pgm":
                                _write_pgm_u16(view_mask_path, view_mask_u16)
                            else:
                                _write_png_depth_u16(view_mask_path, view_mask_u16)
                            view_mask_paths.append(view_mask_path)
                        if self.config.part_segmentation_masks:
                            for view_index, view_part_mask_u16 in enumerate(view_part_masks_u16):
                                view_part_mask_path = per_view_assets_dirs[view_index] / part_mask_name
                                if self.config.mask_format == "pgm":
                                    _write_pgm_u16(view_part_mask_path, view_part_mask_u16)
                                else:
                                    _write_png_depth_u16(view_part_mask_path, view_part_mask_u16)
                                view_part_mask_paths.append(view_part_mask_path)
                else:
                    mask_path = None

                if self.config.camera_mode != "triview" and self.config.part_segmentation_masks and part_mask_path is not None:
                    view_part_mask_paths = [part_mask_path]

                primary_rgb_path = _primary_view_path(view_rgb_paths, rgb_path)
                primary_depth_path = _primary_view_path(view_depth_paths, depth_path)
                primary_mask_path = _primary_view_path(view_mask_paths, mask_path)
                primary_part_mask_path = _primary_view_path(view_part_mask_paths, part_mask_path)

                if concat_video_writer is not None:
                    concat_video_writer.append_data(rgb)
                if view_video_writers:
                    for writer, view_rgb in zip(view_video_writers, rgb_views):
                        writer.append_data(view_rgb)

                frame_payload.append(
                    {
                        "timestamp_s": round(frame_index * self.config.frame_dt, 6),
                        "rgb_path": str(Path(primary_rgb_path).relative_to(output_dir)),
                        "depth_path": str(Path(primary_depth_path).relative_to(output_dir)),
                        "mask_path": (
                            str(Path(primary_mask_path).relative_to(output_dir))
                            if write_binary_masks and primary_mask_path is not None
                            else None
                        ),
                        "part_mask_path": (
                            str(Path(primary_part_mask_path).relative_to(output_dir))
                            if self.config.part_segmentation_masks and primary_part_mask_path is not None
                            else None
                        ),
                        "rgb_paths_by_view": (
                            [str(path.relative_to(output_dir)) for path in view_rgb_paths]
                            if view_rgb_paths
                            else [str(Path(primary_rgb_path).relative_to(output_dir))]
                        ),
                        "depth_paths_by_view": (
                            [str(path.relative_to(output_dir)) for path in view_depth_paths]
                            if view_depth_paths
                            else [str(Path(primary_depth_path).relative_to(output_dir))]
                        ),
                        "mask_paths_by_view": (
                            [str(path.relative_to(output_dir)) for path in view_mask_paths]
                            if view_mask_paths
                            else (
                                [str(Path(primary_mask_path).relative_to(output_dir))]
                                if write_binary_masks and primary_mask_path is not None
                                else []
                            )
                        ),
                        "part_mask_paths_by_view": (
                            [str(path.relative_to(output_dir)) for path in view_part_mask_paths]
                            if view_part_mask_paths
                            else (
                                [str(Path(primary_part_mask_path).relative_to(output_dir))]
                                if self.config.part_segmentation_masks and primary_part_mask_path is not None
                                else []
                            )
                        ),
                        "camera_pose": view_camera_poses[len(view_camera_poses) // 2],
                        "camera_poses_by_view": view_camera_poses,
                        "action_log": {
                            "target_open_fraction": float(target_fraction),
                            "target_joint_position": float(target_q),
                            "command_delta": float(target_q - previous_target),
                            "applied_force_mean": (
                                float(sum(force_history) / len(force_history)) if force_history else 0.0
                            ),
                            "excitation_force_mean": (
                                float(sum(excitation_force_history) / len(excitation_force_history))
                                if excitation_force_history
                                else 0.0
                            ),
                            "excitation_mode": self.config.excitation_mode,
                            "joint_positions": {
                                name: float(data.qpos[qpos_adr])
                                for name, (_, qpos_adr, _) in zip(joint_names, controlled)
                            },
                        },
                        "joint_position_hint": float(data.qpos[primary_qpos_adr]),
                        "observation_confidence": 1.0,
                    }
                )
                previous_target = target_q
        finally:
            if concat_video_writer is not None:
                concat_video_writer.close()
            for writer in view_video_writers:
                writer.close()

        episode_payload = {
            "object_instance_id": self.config.object_instance_id,
            "category": self.config.category,
            "camera_intrinsics": _intrinsics_from_fovy(
                width=self.config.width,
                height=self.config.height,
                fovy_deg=self.config.camera_fovy_deg,
            ),
            "frames": frame_payload,
            "metadata": {
                "source": "mujoco-recorder",
                "model_path": str(Path(self.config.model_path).resolve()),
                "joint_name": self._joint_name(mujoco, model, primary_joint_id),
                "joint_limits": [float(joint_limits[0]), float(joint_limits[1])],
                "joint_names": list(joint_names),
                "all_joints": bool(self.config.all_joints),
                "sim_dt": float(self.config.sim_dt),
                "frame_dt": float(self.config.frame_dt),
                "seed": int(self.config.seed),
                "perturbation_scale": float(self.config.perturbation_scale),
                "control_mode": self.config.control_mode,
                "recording_variant": (
                    "forced-excitation" if self.config.excitation_mode != "none" else self.config.control_mode
                ),
                "random_initial_qpos": bool(self.config.random_initial_qpos),
                "auto_initial_qvel_from_limits": bool(self.config.auto_initial_qvel_from_limits),
                "auto_initial_qvel_direction_mode": self.config.auto_initial_qvel_direction_mode,
                "auto_initial_qvel_min_abs": float(self.config.auto_initial_qvel_min_abs),
                "auto_initial_qvel_max_abs": float(self.config.auto_initial_qvel_max_abs),
                "initial_joint_qvel": float(self.config.initial_joint_qvel),
                "staged_initial_qvel": {
                    "enabled": bool(self.config.staged_initial_qvel),
                    "primary_tokens": list(self.config.staged_primary_tokens),
                    "secondary_tokens": list(self.config.staged_secondary_tokens),
                    "secondary_count_min": int(self.config.staged_secondary_count_min),
                    "secondary_count_max": int(self.config.staged_secondary_count_max),
                    "secondary_start_s": float(self.config.staged_secondary_start_s),
                    "selected_secondary_joints": [
                        self._joint_name(mujoco, model, joint_id) for joint_id in staged_secondary_joints
                    ],
                },
                "kick": {
                    "force": float(self.config.kick_force),
                    "start_s": float(self.config.kick_start_s),
                    "duration_s": float(self.config.kick_duration_s),
                },
                "excitation": {
                    "mode": self.config.excitation_mode,
                    "force": float(self.config.excitation_force),
                    "start_s": float(self.config.excitation_start_s),
                    "duration_s": float(self.config.excitation_duration_s),
                    "period_s": float(self.config.excitation_period_s),
                    "frequency_hz": float(self.config.excitation_frequency_hz),
                    "target": "all_joints" if self.config.all_joints else "selected_joint",
                    "dynamics_log_path": "dynamics_log.jsonl" if self.config.write_dynamics_log else None,
                },
                "rgb_format": self.config.rgb_format,
                "depth_format": self.config.depth_format,
                "video": {
                    "enabled": bool(self.config.make_video),
                    "fps": (
                        float(self.config.video_fps)
                        if self.config.video_fps is not None
                        else float(1.0 / self.config.frame_dt)
                    ),
                    "path": (
                        "episode_concat.mp4"
                        if self.config.make_video and self.config.camera_mode == "triview"
                        else ("episode.mp4" if self.config.make_video else None)
                    ),
                    "paths_by_view": (
                        [f"episode_view_{index}.mp4" for index in range(3)]
                        if self.config.make_video and self.config.camera_mode == "triview"
                        else []
                    ),
                },
                "camera_mode": self.config.camera_mode,
                "camera_pose_convention": "mujoco-gl-forward",
                "depth_convention": "z-depth",
                "camera_distance": float(self.config.camera_distance),
                "camera_elevation_deg": float(self.config.camera_elevation_deg),
                "camera_fovy_deg": float(self.config.camera_fovy_deg),
                "lookat": [float(v) for v in self.config.lookat],
                "segmentation_masks": bool(self.config.segmentation_masks or self.config.part_segmentation_masks),
                "part_segmentation_masks": bool(self.config.part_segmentation_masks),
                "mask_format": self.config.mask_format,
                "triview_asset_layout": (
                    "concat+views" if write_concat_assets and self.config.camera_mode == "triview" else "views-only"
                ),
                "part_segmentation": part_segmentation,
                "disable_target_mesh_collision": bool(self.config.disable_target_mesh_collision),
                "hide_clear_meshes": bool(self.config.hide_clear_meshes),
                "hidden_clear_geom_ids": [int(geom_id) for geom_id in sorted(hidden_clear_geom_ids)],
                "target_root_body_id": int(target_root_body_id),
                "target_root_body_name": self._body_name(mujoco, model, target_root_body_id),
                "target_geom_ids": [int(geom_id) for geom_id in sorted(target_geom_ids)],
                "camera_triview_spacing_deg": float(self.config.camera_triview_spacing_deg),
                "camera_azimuths_deg": (
                    [float(a) for a in triview_azimuths]
                    if self.config.camera_mode == "triview"
                    else [float(self.config.camera_azimuth_start_deg), float(self.config.camera_azimuth_start_deg + self.config.camera_azimuth_span_deg)]
                ),
            },
        }

        episode_path = output_dir / "episode.json"
        if self.config.write_dynamics_log:
            dynamics_log_path = output_dir / "dynamics_log.jsonl"
            dynamics_log_path.write_text(
                "".join(f"{self._json_dumps(row)}\n" for row in dynamics_log_rows),
                encoding="utf-8",
            )
        save_json(episode_payload, episode_path)
        return episode_path

    def _json_dumps(self, value: Any) -> str:
        import json

        return json.dumps(value, separators=(",", ":"), sort_keys=True)

    def _excitation_force(self, timestamp_s: float, joint_index: int) -> float:
        mode = self.config.excitation_mode
        if mode == "none":
            return 0.0
        start_s = float(self.config.excitation_start_s)
        duration_s = float(self.config.excitation_duration_s)
        if timestamp_s < start_s or timestamp_s >= start_s + duration_s:
            return 0.0

        amplitude = float(self.config.excitation_force)
        elapsed_s = timestamp_s - start_s
        if mode == "pulse":
            return amplitude
        if mode == "sine":
            return amplitude * math.sin(2.0 * math.pi * float(self.config.excitation_frequency_hz) * elapsed_s)
        if mode == "prbs":
            segment_index = int(math.floor(elapsed_s / float(self.config.excitation_period_s)))
            rng = random.Random(int(self.config.seed) + 1000003 * int(joint_index) + 9176 * segment_index)
            return amplitude if rng.random() >= 0.5 else -amplitude
        return 0.0

    def _initialize_free_joint_state(
        self,
        mujoco: Any,
        model: Any,
        data: Any,
        joint_id: int,
        qpos_adr: int,
        dof_adr: int,
        primary_joint_id: int,
        joint_index: int,
        rng: random.Random,
    ) -> None:
        lower, upper = self._joint_limits(model, joint_id)
        if not (math.isfinite(lower) and math.isfinite(upper) and upper > lower):
            data.qvel[dof_adr] = float(self.config.initial_joint_qvel)
            return

        span = upper - lower
        forced_response = self.config.excitation_mode != "none"

        # Free-decay recordings move slightly away from hard limits to avoid
        # immediate clipping. Forced-response recordings keep the authored
        # initial state unless random initialization is explicitly requested.
        limit_margin = 0.02 * span
        rest_q = float(data.qpos[qpos_adr])
        min_q = lower + limit_margin
        max_q = upper - limit_margin
        if min_q < max_q:
            default_q = float(model.qpos0[qpos_adr])
            clamped_default_q = min(max(default_q, min_q), max_q)
            if self.config.random_initial_qpos:
                if self.config.auto_initial_qvel_from_limits and joint_id == primary_joint_id:
                    data.qpos[qpos_adr] = clamped_default_q
                else:
                    data.qpos[qpos_adr] = rng.uniform(min_q, max_q)
            elif forced_response:
                data.qpos[qpos_adr] = min(max(rest_q, lower), upper)
            else:
                data.qpos[qpos_adr] = min(max(rest_q, min_q), max_q)

        desired_speed = 0.25 * span / self.config.frame_dt
        if self.config.staged_initial_qvel:
            if self._joint_matches_tokens(mujoco, model, joint_id, self.config.staged_primary_tokens):
                data.qvel[dof_adr] = self._free_initial_qvel(
                    model,
                    data,
                    joint_id,
                    qpos_adr,
                    joint_index,
                    desired_speed,
                    lower,
                    upper,
                )
            else:
                data.qvel[dof_adr] = 0.0
        elif forced_response:
            data.qvel[dof_adr] = float(self.config.initial_joint_qvel)
        elif self.config.auto_initial_qvel_from_limits:
            data.qvel[dof_adr] = self._free_initial_qvel(
                model,
                data,
                joint_id,
                qpos_adr,
                joint_index,
                desired_speed,
                lower,
                upper,
            )
        elif self.config.initial_joint_qvel != 0.0:
            data.qvel[dof_adr] = float(self.config.initial_joint_qvel)
        else:
            direction = 1.0 if (joint_index % 2 == 0) else -1.0
            data.qvel[dof_adr] = direction * desired_speed

    def _free_initial_qvel(
        self,
        model: Any,
        data: Any,
        joint_id: int,
        qpos_adr: int,
        joint_index: int,
        desired_speed: float,
        lower: float,
        upper: float,
    ) -> float:
        if not self.config.auto_initial_qvel_from_limits:
            if self.config.initial_joint_qvel != 0.0:
                return float(self.config.initial_joint_qvel)
            direction = 1.0 if (joint_index % 2 == 0) else -1.0
            return direction * float(desired_speed)

        q0 = float(data.qpos[qpos_adr])
        qref = float(model.qpos0[qpos_adr])
        ref_room_upper = max(0.0, upper - qref)
        ref_room_lower = max(0.0, qref - lower)

        if self.config.auto_initial_qvel_direction_mode == "toward-upper":
            direction = 1.0
        elif self.config.auto_initial_qvel_direction_mode == "toward-lower":
            direction = -1.0
        else:
            direction = 1.0 if ref_room_upper > ref_room_lower else -1.0
        room = max(0.0, (upper - q0) if direction > 0.0 else (q0 - lower))

        effective_min_abs = min(
            float(self.config.auto_initial_qvel_min_abs),
            float(self.config.auto_initial_qvel_max_abs),
        )
        base_speed = max(float(desired_speed), effective_min_abs)
        span = upper - lower
        room_ratio = room / span if span > 1e-9 else 0.0
        speed = base_speed * (0.25 + 0.75 * room_ratio)
        speed = min(speed, float(self.config.auto_initial_qvel_max_abs))
        return direction * speed

    def _open_video_writer(self, output_dir: Path, fps: float, filename: str):
        try:
            import imageio.v2 as imageio
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Video writing requires imageio. Install with: pip install -e '.[viz]'"
            ) from exc

        video_path = output_dir / filename
        return imageio.get_writer(video_path, fps=float(fps), codec="libx264")

    def _camera_pose_from_scene_camera(self, scene_camera) -> list[list[float]]:
        forward = _normalize([float(value) for value in scene_camera.forward])
        up = _normalize([float(value) for value in scene_camera.up])
        right = _normalize(_cross(forward, up))
        up = _normalize(_cross(right, forward))
        position = [float(value) for value in scene_camera.pos]
        return [
            [right[0], up[0], forward[0], position[0]],
            [right[1], up[1], forward[1], position[1]],
            [right[2], up[2], forward[2], position[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]

    def _import_mujoco(self):
        try:
            import mujoco  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mujoco is required for record-mujoco. Install with: pip install '.[simulation]'"
            ) from exc
        return mujoco

    def _select_single_dof_joint(self, mujoco, model) -> int:
        allowed = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        candidates = [joint_id for joint_id in range(model.njnt) if int(model.jnt_type[joint_id]) in allowed]
        if not candidates:
            raise ValueError("No hinge/slide joint found in model for single-DOF recording")

        ranked = sorted(
            candidates,
            key=lambda joint_id: self._single_joint_priority(mujoco, model, joint_id),
            reverse=True,
        )
        return ranked[0]

    def _single_joint_priority(self, mujoco, model, joint_id: int) -> tuple[int, int, float, int]:
        joint_type = int(model.jnt_type[joint_id])
        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
        slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)

        if self.config.category in PRISMATIC_CATEGORIES:
            type_score = 2 if joint_type == slide_type else 1
        elif self.config.category in HINGE_CATEGORIES or self.config.category in FULL_ROTATION_CATEGORIES:
            type_score = 2 if joint_type == hinge_type else 1
        else:
            type_score = 1

        name_score = self._joint_name_score(mujoco, model, joint_id)
        span = self._joint_range_span(model, joint_id)
        span_score = -abs(span - self._target_joint_span_rad(joint_type))

        # Earlier joints remain a mild tiebreaker so selection stays deterministic.
        index_score = -joint_id
        return type_score, name_score, span_score, index_score

    def _joint_name_score(self, mujoco, model, joint_id: int) -> int:
        joint_name = self._joint_name(mujoco, model, joint_id).lower()
        body_name = self._body_name(mujoco, model, int(model.jnt_bodyid[joint_id])).lower()
        text = f"{joint_name} {body_name}"

        positive_tokens: tuple[str, ...]
        negative_tokens = ("disc", "turntable", "tray", "knob", "button", "switch", "wheel")
        if self.config.category in PRISMATIC_CATEGORIES:
            positive_tokens = ("drawer", "slide", "shelf", "rack")
        elif self.config.category in HINGE_CATEGORIES:
            positive_tokens = ("door", "lid", "cover", "flap")
        else:
            positive_tokens = ("door", "lid", "drawer", "cover", "flap")

        score = 0
        for token in positive_tokens:
            if token in text:
                score += 4
        for token in negative_tokens:
            if token in text:
                score -= 4
        return score

    def _joint_range_span(self, model, joint_id: int) -> float:
        if bool(model.jnt_limited[joint_id]):
            lower = float(model.jnt_range[joint_id][0])
            upper = float(model.jnt_range[joint_id][1])
            if upper > lower:
                return upper - lower
        return self._target_joint_span_rad(int(model.jnt_type[joint_id]))

    def _target_joint_span_rad(self, joint_type: int) -> float:
        # Prefer door-like partial hinges for hinge-centric categories and short slides
        # for prismatic categories. Full-turn categories bias toward larger rotary spans.
        slide_type = 2
        if self.config.category in FULL_ROTATION_CATEGORIES:
            return 6.283185307179586
        if self.config.category in PRISMATIC_CATEGORIES:
            return 0.35
        if self.config.category in HINGE_CATEGORIES:
            return 1.5708
        return 0.35 if joint_type == slide_type else 1.5708

    def _select_controlled_joints(self, mujoco, model) -> list[tuple[int, int, int]]:
        allowed = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}

        if self.config.all_joints or self.config.staged_initial_qvel:
            joint_ids = [jid for jid in range(model.njnt) if int(model.jnt_type[jid]) in allowed]
            if not joint_ids:
                raise ValueError("No hinge/slide joint found in model")
            return [
                (jid, int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
                for jid in joint_ids
            ]

        joint_id = self._resolve_joint_id(mujoco, model)
        return [(joint_id, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]))]

    def _select_staged_secondary_joints(
        self,
        mujoco,
        model,
        controlled: list[tuple[int, int, int]],
        rng: random.Random,
    ) -> set[int]:
        if not self.config.staged_initial_qvel:
            return set()
        candidates = [
            joint_id
            for joint_id, _qpos_adr, _dof_adr in controlled
            if self._joint_matches_tokens(mujoco, model, joint_id, self.config.staged_secondary_tokens)
        ]
        if not candidates:
            return set()
        min_count = max(0, int(self.config.staged_secondary_count_min))
        max_count = max(min_count, int(self.config.staged_secondary_count_max))
        count = min(len(candidates), rng.randint(min_count, max_count) if max_count > 0 else 0)
        if count <= 0:
            return set()
        return set(rng.sample(candidates, count))

    def _joint_matches_tokens(self, mujoco, model, joint_id: int, tokens: tuple[str, ...]) -> bool:
        if not tokens:
            return False
        body_id = int(model.jnt_bodyid[joint_id])
        text = f"{self._joint_name(mujoco, model, joint_id)} {self._body_name(mujoco, model, body_id)}".lower()
        return any(str(token).lower() in text for token in tokens)

    def _resolve_joint_id(self, mujoco, model) -> int:
        allowed = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}

        joint_id: int
        if self.config.joint_id is not None:
            joint_id = int(self.config.joint_id)
            if joint_id < 0 or joint_id >= model.njnt:
                raise ValueError(f"joint_id out of range: {joint_id}")
        elif self.config.joint_name is not None:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, self.config.joint_name))
            if joint_id < 0:
                raise ValueError(f"Unknown joint_name: {self.config.joint_name}")
        else:
            joint_id = self._select_single_dof_joint(mujoco, model)

        if int(model.jnt_type[joint_id]) not in allowed:
            name = self._joint_name(mujoco, model, joint_id)
            raise ValueError(f"Selected joint '{name}' is not hinge/slide")
        return joint_id

    def _joint_limits(self, model, joint_id: int) -> tuple[float, float]:
        if bool(model.jnt_limited[joint_id]):
            lower = float(model.jnt_range[joint_id][0])
            upper = float(model.jnt_range[joint_id][1])
            if upper > lower:
                return lower, upper
        if self.config.category in FULL_ROTATION_CATEGORIES:
            return 0.0, 6.283185307179586
        if self.config.category in HINGE_CATEGORIES:
            return 0.0, 1.5708
        return 0.0, 0.35

    def _joint_name(self, mujoco, model, joint_id: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        return name if name else f"joint_{joint_id}"

    def _body_name(self, mujoco, model, body_id: int) -> str:
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        return name if name else f"body_{body_id}"

    def _infer_object_root_body_id(self, model, controlled: list[tuple[int, int, int]]) -> int:
        body_ids = [int(model.jnt_bodyid[joint_id]) for joint_id, _, _ in controlled]
        if not body_ids:
            return 1 if model.nbody > 1 else 0

        ancestor_paths: list[list[int]] = []
        for body_id in body_ids:
            path = [body_id]
            current = body_id
            while current > 0:
                current = int(model.body_parentid[current])
                path.append(current)
            ancestor_paths.append(path)

        common = set(ancestor_paths[0])
        for path in ancestor_paths[1:]:
            common &= set(path)
        if not common:
            return 1 if model.nbody > 1 else 0

        candidates = [body_id for body_id in common if body_id != 0]
        if not candidates:
            return 1 if model.nbody > 1 else 0

        def depth(body_id: int) -> int:
            current = body_id
            count = 0
            while current > 0:
                current = int(model.body_parentid[current])
                count += 1
            return count

        return min(candidates, key=depth)

    def _collect_descendant_body_ids(self, model, root_body_id: int) -> set[int]:
        descendants = {int(root_body_id)}
        changed = True
        while changed:
            changed = False
            for body_id in range(model.nbody):
                parent_id = int(model.body_parentid[body_id])
                if parent_id in descendants and body_id not in descendants:
                    descendants.add(int(body_id))
                    changed = True
        return descendants

    def _collect_target_geom_ids(self, model, root_body_id: int) -> set[int]:
        body_ids = self._collect_descendant_body_ids(model, root_body_id)
        target_geom_ids: set[int] = set()
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) in body_ids:
                target_geom_ids.add(int(geom_id))
        return target_geom_ids

    def _disable_target_mesh_collision(self, model, target_geom_ids: set[int]) -> None:
        for geom_id in target_geom_ids:
            if int(model.geom_type[geom_id]) != 7:
                continue
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0

    def _collect_clear_mesh_geom_ids(self, mujoco, model, target_geom_ids: set[int]) -> set[int]:
        clear_geom_ids: set[int] = set()
        for geom_id in target_geom_ids:
            if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
                continue
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            body_name = self._body_name(mujoco, model, int(model.geom_bodyid[geom_id]))
            mesh_id = int(model.geom_dataid[geom_id])
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
            text = f"{geom_name} {body_name} {mesh_name}".lower()
            if "clear" in text:
                clear_geom_ids.add(int(geom_id))
        return clear_geom_ids

    def _scene_option_hiding_geom_ids(self, mujoco, model, geom_ids: set[int]):
        if not geom_ids:
            return None
        hidden_group = 5
        for geom_id in geom_ids:
            model.geom_group[geom_id] = hidden_group
        option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(option)
        for group_index in range(len(option.geomgroup)):
            option.geomgroup[group_index] = 1
        option.geomgroup[hidden_group] = 0
        return option

    def _segmentation_to_binary_mask_u16(self, mujoco, segmentation, target_geom_ids: set[int]):
        import numpy as np

        if segmentation.ndim != 3 or segmentation.shape[2] < 2:
            raise ValueError("Expected MuJoCo segmentation render with at least 2 channels")
        geom_ids = segmentation[:, :, 0].astype(np.int32, copy=False)
        object_types = segmentation[:, :, 1].astype(np.int32, copy=False)
        geom_type_id = int(mujoco.mjtObj.mjOBJ_GEOM)
        mask = np.isin(geom_ids, np.array(sorted(target_geom_ids), dtype=np.int32))
        mask &= object_types == geom_type_id
        return (mask.astype(np.uint16) * np.uint16(65535))

    def _binary_mask_from_part_mask_u16(self, part_mask_u16):
        import numpy as np

        return (np.asarray(part_mask_u16, dtype=np.uint16) > 0).astype(np.uint16) * np.uint16(65535)


class MuJoCoMaskRenderer:
    def __init__(self, config: MuJoCoMaskRenderConfig) -> None:
        self.config = config

    def render(self) -> Path:
        if self.config.mask_format not in {"pgm", "png"}:
            raise ValueError("mask_format must be 'pgm' or 'png'")

        episode_path = Path(self.config.episode_path).resolve()
        episode_payload = load_json(episode_path)
        output_episode_path = (
            Path(self.config.output_episode_path).resolve()
            if self.config.output_episode_path is not None
            else episode_path
        )
        output_dir = output_episode_path.parent

        mujoco = self._import_mujoco()
        import numpy as np

        model_path = (
            Path(self.config.model_path).resolve()
            if self.config.model_path is not None
            else Path(episode_payload["metadata"]["model_path"]).resolve()
        )
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)

        first_frame = episode_payload["frames"][0]
        first_depth_path = first_frame.get("depth_paths_by_view", [first_frame["depth_path"]])[0]
        depth_height, depth_width = self._read_image_shape(output_dir / first_depth_path)
        camera_fovy_deg = episode_payload.get("metadata", {}).get("camera_fovy_deg")
        if camera_fovy_deg is None:
            camera_fovy_deg = _fovy_deg_from_intrinsics(
                height=depth_height,
                camera_intrinsics=dict(episode_payload.get("camera_intrinsics", {})),
            )
        if camera_fovy_deg is not None:
            model.vis.global_.fovy = float(camera_fovy_deg)
        renderer = mujoco.Renderer(model, width=depth_width, height=depth_height)

        joint_names = episode_payload.get("metadata", {}).get("joint_names", [])
        controlled: list[tuple[int, int, int]] = []
        for joint_name in joint_names:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            if joint_id >= 0:
                controlled.append((joint_id, int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])))
        recorder = MuJoCoEpisodeRecorder(
            MuJoCoRecordConfig(
                model_path=model_path,
                output_dir=output_dir,
                object_instance_id=episode_payload["object_instance_id"],
                category=episode_payload["category"],
            )
        )
        target_root_body_id = int(
            episode_payload.get("metadata", {}).get(
                "target_root_body_id",
                recorder._infer_object_root_body_id(model, controlled) if controlled else 1,
            )
        )
        target_geom_ids = recorder._collect_target_geom_ids(model, target_root_body_id)
        hidden_clear_geom_ids: set[int] = set()
        scene_option = None
        if bool(episode_payload.get("metadata", {}).get("hide_clear_meshes", False)):
            hidden_clear_geom_ids = recorder._collect_clear_mesh_geom_ids(mujoco, model, target_geom_ids)
            scene_option = recorder._scene_option_hiding_geom_ids(mujoco, model, hidden_clear_geom_ids)
        write_part_masks = bool(
            self.config.part_segmentation_masks or episode_payload.get("metadata", {}).get("part_segmentation")
        )
        part_segmentation = (
            build_mujoco_body_part_segmentation(
                mujoco=mujoco,
                model=model,
                root_body_id=target_root_body_id,
                target_geom_ids=target_geom_ids,
                hidden_geom_ids=hidden_clear_geom_ids,
            )
            if write_part_masks
            else None
        )

        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE

        camera_meta = episode_payload.get("metadata", {})
        camera_mode = camera_meta.get("camera_mode", "orbit")
        pose_convention = camera_meta.get(
            "camera_pose_convention",
            "legacy-mujoco-orbit" if camera_meta.get("source") == "mujoco-recorder" else "camera-to-world-forward",
        )
        lookat = [float(value) for value in camera_meta.get("lookat", [0.0, 0.0, 0.0])]
        camera_distance = float(camera_meta.get("camera_distance", 1.0))
        camera_elevation_deg = float(camera_meta.get("camera_elevation_deg", 0.0))
        camera_azimuths_deg = [float(value) for value in camera_meta.get("camera_azimuths_deg", [])]

        assets_dir = output_dir / "assets"
        write_concat_assets = camera_meta.get("triview_asset_layout", "concat+views") != "views-only"
        concat_assets_dir = assets_dir / "concat" if camera_mode == "triview" else assets_dir
        per_view_assets_dirs = [assets_dir / f"view_{index}" for index in range(3)] if camera_mode == "triview" else []
        if camera_mode != "triview" or write_concat_assets:
            concat_assets_dir.mkdir(parents=True, exist_ok=True)
        for view_dir in per_view_assets_dirs:
            view_dir.mkdir(parents=True, exist_ok=True)

        mask_ext = "pgm" if self.config.mask_format == "pgm" else "png"
        for frame_index, frame in enumerate(episode_payload["frames"]):
            self._apply_joint_positions(mujoco, model, data, frame.get("action_log", {}).get("joint_positions", {}))
            mujoco.mj_forward(model, data)

            if camera_mode == "triview" and len(camera_azimuths_deg) >= 3:
                azimuths = camera_azimuths_deg[:3]
            elif frame.get("camera_poses_by_view"):
                azimuths = self._azimuths_from_poses(frame["camera_poses_by_view"], lookat, str(pose_convention))
            else:
                azimuths = [self._azimuth_from_pose(frame["camera_pose"], lookat, str(pose_convention))]

            view_masks_u16 = []
            view_part_masks_u16 = []
            for azimuth in azimuths:
                camera.azimuth = azimuth
                camera.elevation = camera_elevation_deg
                camera.distance = camera_distance
                camera.lookat[:] = list(lookat)
                renderer.enable_segmentation_rendering()
                renderer.update_scene(data, camera=camera, scene_option=scene_option)
                segmentation = renderer.render()
                renderer.disable_segmentation_rendering()
                if write_part_masks and part_segmentation is not None:
                    part_mask_u16 = segmentation_to_part_mask_u16(
                        mujoco=mujoco,
                        segmentation=segmentation,
                        target_geom_ids=target_geom_ids,
                        part_segmentation=part_segmentation,
                    )
                    view_part_masks_u16.append(part_mask_u16)
                    view_masks_u16.append(recorder._binary_mask_from_part_mask_u16(part_mask_u16))
                else:
                    view_masks_u16.append(
                        recorder._segmentation_to_binary_mask_u16(
                            mujoco=mujoco,
                            segmentation=segmentation,
                            target_geom_ids=target_geom_ids,
                        )
                    )

            mask_name = f"frame_{frame_index:04d}_mask.{mask_ext}"
            part_mask_name = f"frame_{frame_index:04d}_part_mask.{mask_ext}"
            if len(view_masks_u16) == 1:
                mask_u16 = view_masks_u16[0]
            else:
                mask_u16 = np.concatenate(view_masks_u16, axis=1)
            mask_path = None
            if camera_mode != "triview" or write_concat_assets:
                mask_path = concat_assets_dir / mask_name
                if self.config.mask_format == "pgm":
                    _write_pgm_u16(mask_path, mask_u16)
                else:
                    _write_png_depth_u16(mask_path, mask_u16)

            frame["mask_paths_by_view"] = []
            frame["part_mask_path"] = None
            frame["part_mask_paths_by_view"] = []
            if write_part_masks and view_part_masks_u16:
                if len(view_part_masks_u16) == 1:
                    part_mask_u16 = view_part_masks_u16[0]
                else:
                    part_mask_u16 = np.concatenate(view_part_masks_u16, axis=1)
                if camera_mode != "triview" or write_concat_assets:
                    part_mask_path = concat_assets_dir / part_mask_name
                    if self.config.mask_format == "pgm":
                        _write_pgm_u16(part_mask_path, part_mask_u16)
                    else:
                        _write_png_depth_u16(part_mask_path, part_mask_u16)
                    frame["part_mask_path"] = str(part_mask_path.relative_to(output_dir))
            if camera_mode == "triview":
                for view_index, view_mask_u16 in enumerate(view_masks_u16):
                    view_mask_path = per_view_assets_dirs[view_index] / mask_name
                    if self.config.mask_format == "pgm":
                        _write_pgm_u16(view_mask_path, view_mask_u16)
                    else:
                        _write_png_depth_u16(view_mask_path, view_mask_u16)
                    frame["mask_paths_by_view"].append(str(view_mask_path.relative_to(output_dir)))
                if write_part_masks:
                    for view_index, view_part_mask_u16 in enumerate(view_part_masks_u16):
                        view_part_mask_path = per_view_assets_dirs[view_index] / part_mask_name
                        if self.config.mask_format == "pgm":
                            _write_pgm_u16(view_part_mask_path, view_part_mask_u16)
                        else:
                            _write_png_depth_u16(view_part_mask_path, view_part_mask_u16)
                        frame["part_mask_paths_by_view"].append(str(view_part_mask_path.relative_to(output_dir)))
                frame["mask_path"] = str(Path(_primary_view_path(frame["mask_paths_by_view"], mask_path)).as_posix())
                if frame["part_mask_paths_by_view"]:
                    frame["part_mask_path"] = str(
                        Path(_primary_view_path(frame["part_mask_paths_by_view"], frame["part_mask_path"])).as_posix()
                    )
            else:
                frame["mask_path"] = str(mask_path.relative_to(output_dir))
                frame["mask_paths_by_view"] = [str(mask_path.relative_to(output_dir))]
                if frame["part_mask_path"] is not None:
                    frame["part_mask_paths_by_view"] = [str(frame["part_mask_path"])]

        episode_payload.setdefault("metadata", {})
        episode_payload["metadata"]["segmentation_masks"] = True
        episode_payload["metadata"]["part_segmentation_masks"] = bool(write_part_masks)
        episode_payload["metadata"]["mask_format"] = self.config.mask_format
        episode_payload["metadata"]["triview_asset_layout"] = (
            "concat+views" if write_concat_assets and camera_mode == "triview" else "views-only"
        )
        episode_payload["metadata"]["depth_convention"] = "z-depth"
        if camera_fovy_deg is not None:
            episode_payload["metadata"]["camera_fovy_deg"] = float(camera_fovy_deg)
        episode_payload["metadata"]["hide_clear_meshes"] = bool(
            episode_payload["metadata"].get("hide_clear_meshes", False)
        )
        episode_payload["metadata"]["part_segmentation"] = part_segmentation
        episode_payload["metadata"]["hidden_clear_geom_ids"] = [int(geom_id) for geom_id in sorted(hidden_clear_geom_ids)]
        episode_payload["metadata"]["target_root_body_id"] = int(target_root_body_id)
        episode_payload["metadata"]["target_root_body_name"] = recorder._body_name(mujoco, model, target_root_body_id)
        episode_payload["metadata"]["target_geom_ids"] = [int(geom_id) for geom_id in sorted(target_geom_ids)]
        save_json(episode_payload, output_episode_path)
        return output_episode_path

    def _import_mujoco(self):
        try:
            import mujoco  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "mujoco is required for render-mujoco-masks. Install with: pip install '.[simulation]'"
            ) from exc
        return mujoco

    def _apply_joint_positions(self, mujoco, model, data, joint_positions: dict[str, Any]) -> None:
        data.qvel[:] = 0.0
        for joint_name, joint_value in joint_positions.items():
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            if joint_id < 0:
                continue
            qpos_adr = int(model.jnt_qposadr[joint_id])
            data.qpos[qpos_adr] = float(joint_value)

    def _azimuth_from_pose(
        self,
        pose: list[list[float]],
        lookat: list[float],
        pose_convention: str,
    ) -> float:
        if pose_convention == "legacy-mujoco-orbit":
            dx = float(pose[0][3]) - lookat[0]
            dy = float(pose[1][3]) - lookat[1]
            return math.degrees(math.atan2(dy, dx))

        dx = lookat[0] - float(pose[0][3])
        dy = lookat[1] - float(pose[1][3])
        return math.degrees(math.atan2(dy, dx))

    def _azimuths_from_poses(
        self,
        poses: list[list[list[float]]],
        lookat: list[float],
        pose_convention: str,
    ) -> list[float]:
        return [self._azimuth_from_pose(pose, lookat, pose_convention) for pose in poses]

    def _read_image_shape(self, path: Path) -> tuple[int, int]:
        if path.suffix.lower() == ".pgm":
            with path.open("rb") as handle:
                if handle.readline().strip() != b"P5":
                    raise ValueError(f"Unsupported PGM file: {path}")
                tokens: list[bytes] = []
                while len(tokens) < 3:
                    line = handle.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line or line.startswith(b"#"):
                        continue
                    tokens.extend(line.split())
                width = int(tokens[0])
                height = int(tokens[1])
                return height, width

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PNG shape reading requires Pillow.") from exc
        image = Image.open(path)
        width, height = image.size
        return height, width


class MuJoCoEpisodeCompactor:
    def __init__(self, config: MuJoCoEpisodeCompactConfig) -> None:
        self.config = config

    def compact(self) -> dict[str, Any]:
        episode_path = Path(self.config.episode_path).resolve()
        payload = load_json(episode_path)
        metadata = dict(payload.get("metadata", {}))
        frames = list(payload.get("frames", []))
        if metadata.get("camera_mode") != "triview":
            raise ValueError("compact-mujoco-recording currently only supports triview episodes")

        output_dir = episode_path.parent
        concat_dir = output_dir / "assets" / "concat"
        concat_video = output_dir / "episode_concat.mp4"
        removable_bytes = 0
        if self.config.remove_concat_dir:
            removable_bytes += _path_size_bytes(concat_dir)
        if self.config.remove_concat_video:
            removable_bytes += _path_size_bytes(concat_video)

        frame_updates = 0
        for frame in frames:
            rgb_paths = list(frame.get("rgb_paths_by_view", []))
            depth_paths = list(frame.get("depth_paths_by_view", []))
            mask_paths = list(frame.get("mask_paths_by_view", []))
            part_mask_paths = list(frame.get("part_mask_paths_by_view", []))
            if rgb_paths:
                frame["rgb_path"] = str(_primary_view_path(rgb_paths, frame.get("rgb_path")))
            if depth_paths:
                frame["depth_path"] = str(_primary_view_path(depth_paths, frame.get("depth_path")))
            if mask_paths:
                frame["mask_path"] = str(_primary_view_path(mask_paths, frame.get("mask_path")))
            if part_mask_paths:
                frame["part_mask_path"] = str(_primary_view_path(part_mask_paths, frame.get("part_mask_path")))
            frame_updates += 1

        metadata["triview_asset_layout"] = "views-only"
        payload["metadata"] = metadata
        payload["frames"] = frames

        if not self.config.dry_run:
            save_json(payload, episode_path)
            if self.config.remove_concat_dir and concat_dir.is_dir():
                shutil.rmtree(concat_dir)
            if self.config.remove_concat_video and concat_video.is_file():
                concat_video.unlink()

        return {
            "episode_path": str(episode_path),
            "frame_updates": frame_updates,
            "concat_dir": str(concat_dir),
            "concat_dir_exists": concat_dir.exists(),
            "concat_video": str(concat_video),
            "concat_video_exists": concat_video.exists(),
            "dry_run": bool(self.config.dry_run),
            "estimated_bytes_reclaimed": int(removable_bytes),
        }


class MuJoCoEpisodeRepacker:
    def __init__(self, config: MuJoCoEpisodeRepackConfig) -> None:
        self.config = config

    def repack(self) -> dict[str, Any]:
        episode_path = Path(self.config.episode_path).resolve()
        payload = load_json(episode_path)
        frames = list(payload.get("frames", []))
        metadata = dict(payload.get("metadata", {}))
        output_dir = episode_path.parent

        converted_count = 0
        bytes_before = 0
        bytes_after = 0
        bytes_reclaimed = 0
        converted_map: dict[str, str] = {}

        def repack_rel_path(relative_path: str | None, kind: str) -> str | None:
            nonlocal converted_count, bytes_before, bytes_after, bytes_reclaimed
            if not relative_path:
                return relative_path
            if relative_path in converted_map:
                return converted_map[relative_path]

            source = output_dir / relative_path
            suffix = source.suffix.lower()
            if suffix == ".png":
                converted_map[relative_path] = relative_path
                return relative_path

            if suffix not in {".ppm", ".pgm"}:
                converted_map[relative_path] = relative_path
                return relative_path

            target = source.with_suffix(".png")
            if kind == "rgb":
                image = _read_ppm_rgb(source) if suffix == ".ppm" else _read_png_rgb(source)
                encoded_before = int(source.stat().st_size)
                if not self.config.dry_run:
                    _write_png_rgb(target, image)
            else:
                image = _read_pgm_u16(source) if suffix == ".pgm" else _read_png_u16(source)
                encoded_before = int(source.stat().st_size)
                if not self.config.dry_run:
                    _write_png_depth_u16(target, image)

            encoded_after = int(target.stat().st_size) if target.exists() else 0
            bytes_before += encoded_before
            bytes_after += encoded_after
            if not self.config.dry_run and not self.config.keep_originals and source.exists():
                source.unlink()
                bytes_reclaimed += max(0, encoded_before - encoded_after)

            new_relative = str(target.relative_to(output_dir))
            converted_map[relative_path] = new_relative
            converted_count += 1
            return new_relative

        for frame in frames:
            frame["rgb_path"] = repack_rel_path(frame.get("rgb_path"), "rgb")
            frame["depth_path"] = repack_rel_path(frame.get("depth_path"), "depth")
            frame["mask_path"] = repack_rel_path(frame.get("mask_path"), "mask")
            frame["part_mask_path"] = repack_rel_path(frame.get("part_mask_path"), "part-mask")
            frame["rgb_paths_by_view"] = [repack_rel_path(path, "rgb") for path in frame.get("rgb_paths_by_view", [])]
            frame["depth_paths_by_view"] = [
                repack_rel_path(path, "depth") for path in frame.get("depth_paths_by_view", [])
            ]
            frame["mask_paths_by_view"] = [
                repack_rel_path(path, "mask") for path in frame.get("mask_paths_by_view", [])
            ]
            frame["part_mask_paths_by_view"] = [
                repack_rel_path(path, "part-mask") for path in frame.get("part_mask_paths_by_view", [])
            ]

        metadata["rgb_format"] = "png"
        metadata["depth_format"] = "png"
        if metadata.get("segmentation_masks") or metadata.get("part_segmentation_masks"):
            metadata["mask_format"] = "png"
        payload["metadata"] = metadata
        payload["frames"] = frames

        if not self.config.dry_run:
            save_json(payload, episode_path)

        return {
            "episode_path": str(episode_path),
            "converted_paths": int(converted_count),
            "dry_run": bool(self.config.dry_run),
            "keep_originals": bool(self.config.keep_originals),
            "input_bytes": int(bytes_before),
            "output_bytes": int(bytes_after),
            "estimated_bytes_reclaimed": int(bytes_before if self.config.dry_run and not self.config.keep_originals else bytes_reclaimed),
        }


def _path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total
