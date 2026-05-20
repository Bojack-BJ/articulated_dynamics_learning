from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Project wrapper around Particulate infer.py")
    parser.add_argument("--num_points_global", type=int, default=40000)
    known, remaining = parser.parse_known_args()

    particulate_root = Path.cwd().resolve()
    infer_path = particulate_root / "infer.py"
    if not infer_path.exists():
        raise FileNotFoundError(f"Expected to run from the Particulate root containing infer.py, got {particulate_root}")

    if str(particulate_root) not in sys.path:
        sys.path.insert(0, str(particulate_root))
    partfield_root = particulate_root / "PartField"
    if str(partfield_root) not in sys.path:
        sys.path.insert(0, str(partfield_root))

    module = _load_infer_module(infer_path)

    def prepare_inputs_with_global_points(*args: Any, **kwargs: Any) -> Any:
        return _prepare_inputs_with_timing(module, max(1, int(known.num_points_global)), *args, **kwargs)

    module.prepare_inputs = prepare_inputs_with_global_points
    module.predict_mesh = _make_predict_mesh_with_timing(module)
    module_args = module.argparse.ArgumentParser(description="Particulate Inference Script")
    module_args.add_argument("--input_mesh", type=str, required=True, help="Path to input mesh (.obj, .ply, or .glb)")
    module_args.add_argument("--output_dir", type=str, default="inference_outputs", help="Directory to save outputs")
    module_args.add_argument("--model_config", type=str, default="configs/particulate-B.yaml", help="Path to model config")
    module_args.add_argument("--ckpt_path", type=str, default=None, help="Path to model checkpoint")
    module_args.add_argument(
        "--up_dir",
        type=str,
        default="-Z",
        choices=["X", "Y", "Z", "-X", "-Y", "-Z"],
        help="Up direction of the input mesh",
    )
    module_args.add_argument("--num_points", type=int, default=102400, help="Number of decode points to sample")
    module_args.add_argument("--min_part_confidence", type=float, default=0.0, help="Minimum part confidence")
    module_args.add_argument("--no_strict", action="store_true", help="Disable strict connected component refinement")
    module_args.add_argument("--animation_frames", type=int, default=50, help="Number of animation frames")
    module_args.add_argument("--export_urdf", action="store_true", help="Export URDF")
    module_args.add_argument("--export_mjcf", action="store_true", help="Export MJCF")
    module_args.add_argument("--eval", action="store_true", help="Save results for evaluation")
    module.main(module_args.parse_args(remaining))
    return 0


def _load_infer_module(infer_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("particulate_infer_impl", infer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Particulate infer.py from {infer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_inputs_with_timing(module: Any, num_points_global: int, mesh: Any, *args: Any, **kwargs: Any) -> Any:
    num_points_decode = int(kwargs.get("num_points_decode", 2048))
    device = str(kwargs.get("device", "cuda"))
    sharp_point_ratio = module.DATA_CONFIG["sharp_point_ratio"]
    started = time.perf_counter()

    print(
        "[particulate-timing] prepare_inputs start "
        f"faces={len(mesh.faces)} verts={len(mesh.vertices)} "
        f"global_points={num_points_global} decode_points={num_points_decode}",
        flush=True,
    )

    stage = time.perf_counter()
    all_points, _, _, _ = module.sample_points(mesh, num_points_global, sharp_point_ratio)
    print(f"[particulate-timing] sample_global_s={time.perf_counter() - stage:.2f}", flush=True)

    stage = time.perf_counter()
    points, normals, sharp_flag, face_indices = module.sample_points(mesh, num_points_decode, sharp_point_ratio)
    print(f"[particulate-timing] sample_decode_s={time.perf_counter() - stage:.2f}", flush=True)

    stage = time.perf_counter()
    if module.DATA_CONFIG["normalize_points"]:
        bbmin = module.np.concatenate([all_points, points], axis=0).min(0)
        bbmax = module.np.concatenate([all_points, points], axis=0).max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 1.0 / (bbmax - bbmin).max()
        all_points = (all_points - center) * scale
        points = (points - center) * scale

    all_points = module.torch.from_numpy(all_points).to(device).float().unsqueeze(0)
    points = module.torch.from_numpy(points).to(device).float().unsqueeze(0)
    normals = module.torch.from_numpy(normals).to(device).float().unsqueeze(0)
    print(f"[particulate-timing] tensorize_s={time.perf_counter() - stage:.2f}", flush=True)

    stage = time.perf_counter()
    partfield_model = module.get_partfield_model(device=device)
    print(f"[particulate-timing] load_partfield_s={time.perf_counter() - stage:.2f}", flush=True)

    stage = time.perf_counter()
    feats = module.obtain_partfield_feats(partfield_model, all_points, points)
    print(f"[particulate-timing] obtain_partfield_feats_s={time.perf_counter() - stage:.2f}", flush=True)
    print(f"[particulate-timing] prepare_inputs_total_s={time.perf_counter() - started:.2f}", flush=True)

    return dict(xyz=points, normals=normals, feats=feats), sharp_flag, face_indices


def _make_predict_mesh_with_timing(module: Any) -> Any:
    def predict_mesh_with_timing(mesh: Any, up_dir: str, model: Any, num_points: int, min_part_confidence: float = 0.0) -> Any:
        started = time.perf_counter()
        mesh_transformed = mesh.copy()
        if up_dir == "X":
            rotation_matrix = module.np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=module.np.float32)
            mesh_transformed.vertices = mesh_transformed.vertices @ rotation_matrix.T
        elif up_dir == "-X":
            rotation_matrix = module.np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=module.np.float32)
            mesh_transformed.vertices = mesh_transformed.vertices @ rotation_matrix.T
        elif up_dir == "Y":
            rotation_matrix = module.np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=module.np.float32)
            mesh_transformed.vertices = mesh_transformed.vertices @ rotation_matrix.T
        elif up_dir == "-Y":
            rotation_matrix = module.np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=module.np.float32)
            mesh_transformed.vertices = mesh_transformed.vertices @ rotation_matrix.T
        elif up_dir == "Z":
            pass
        elif up_dir == "-Z":
            rotation_matrix = module.np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=module.np.float32)
            mesh_transformed.vertices = mesh_transformed.vertices @ rotation_matrix.T
        else:
            raise ValueError(f"Invalid up direction: {up_dir}")

        bbox_min = mesh_transformed.vertices.min(axis=0)
        bbox_max = mesh_transformed.vertices.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        mesh_transformed.vertices -= center
        scale = (bbox_max - bbox_min).max()
        mesh_transformed.vertices /= scale

        stage = time.perf_counter()
        inputs, sharp_flag, face_indices = module.prepare_inputs(
            mesh_transformed,
            num_points_global=40000,
            num_points_decode=num_points,
        )
        print(f"[particulate-timing] predict_prepare_inputs_s={time.perf_counter() - stage:.2f}", flush=True)

        stage = time.perf_counter()
        with module.torch.no_grad():
            outputs = model.infer(
                xyz=inputs["xyz"],
                feats=inputs["feats"],
                normals=inputs["normals"],
                output_all_hyps=True,
                min_part_confidence=min_part_confidence,
            )
        print(f"[particulate-timing] model_infer_s={time.perf_counter() - stage:.2f}", flush=True)
        print(f"[particulate-timing] predict_mesh_total_s={time.perf_counter() - started:.2f}", flush=True)
        return outputs, face_indices, mesh_transformed

    return predict_mesh_with_timing


if __name__ == "__main__":
    raise SystemExit(main())
