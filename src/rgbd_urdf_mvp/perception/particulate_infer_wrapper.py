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
    sharp_edge_cache: dict[int, dict[str, Any]] = {}

    def prepare_inputs_with_global_points(mesh: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("num_points_global", None)
        return _prepare_inputs_with_timing(module, max(1, int(known.num_points_global)), mesh, *args, **kwargs)

    module.sample_points = _make_cached_sample_points(module, sharp_edge_cache)
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


def _prepare_inputs_with_timing(module: Any, configured_num_points_global: int, mesh: Any, *args: Any, **kwargs: Any) -> Any:
    num_points_decode = int(args[1] if len(args) > 1 else kwargs.get("num_points_decode", 2048))
    device = str(args[2] if len(args) > 2 else kwargs.get("device", "cuda"))
    sharp_point_ratio = module.DATA_CONFIG["sharp_point_ratio"]
    started = time.perf_counter()

    print(
        "[particulate-timing] prepare_inputs start "
        f"faces={len(mesh.faces)} verts={len(mesh.vertices)} "
        f"global_points={configured_num_points_global} decode_points={num_points_decode}",
        flush=True,
    )

    stage = time.perf_counter()
    all_points, _, _, _ = module.sample_points(mesh, configured_num_points_global, sharp_point_ratio)
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


def _make_cached_sample_points(module: Any, sharp_edge_cache: dict[int, dict[str, Any]]) -> Any:
    def sample_points_cached(mesh: Any, num_points: int, sharp_point_ratio: float) -> Any:
        started = time.perf_counter()
        np = module.np
        num_points = max(1, int(num_points))
        num_points_sharp_edges = int(num_points * float(sharp_point_ratio))
        num_points_uniform = num_points - num_points_sharp_edges
        if num_points_sharp_edges > 0:
            points_sharp, normals_sharp, edge_indices, sharp_edge_faces = _sample_sharp_points_cached(
                module,
                sharp_edge_cache,
                mesh,
                num_points_sharp_edges,
            )
        else:
            points_sharp = np.zeros((0, 3), dtype=np.float64)
            normals_sharp = np.zeros((0, 3), dtype=np.float64)
            edge_indices = np.zeros((0,), dtype=np.int32)
            sharp_edge_faces = np.zeros((0, 2), dtype=np.int32)

        if len(points_sharp) == 0 and sharp_point_ratio > 0:
            print("[particulate-timing] no_sharp_edges_found; sampling uniformly", flush=True)
            num_points_uniform = num_points

        if num_points_uniform > 0:
            stage = time.perf_counter()
            points_uniform, face_indices = mesh.sample(num_points_uniform, return_index=True)
            normals_uniform = mesh.face_normals[face_indices]
            print(
                f"[particulate-timing] uniform_sample_points={num_points_uniform} "
                f"elapsed_s={time.perf_counter() - stage:.2f}",
                flush=True,
            )
        else:
            points_uniform = np.zeros((0, 3), dtype=np.float64)
            normals_uniform = np.zeros((0, 3), dtype=np.float64)
            face_indices = np.zeros((0,), dtype=np.int32)

        points = np.concatenate([points_sharp, points_uniform], axis=0)
        normals = np.concatenate([normals_sharp, normals_uniform], axis=0)
        sharp_flag = np.concatenate(
            [
                np.ones(len(points_sharp), dtype=np.bool_),
                np.zeros(len(points_uniform), dtype=np.bool_),
            ],
            axis=0,
        )

        sharp_face_indices = np.zeros(len(points_sharp), dtype=np.int32)
        if len(points_sharp) > 0:
            selected_faces = sharp_edge_faces[edge_indices]
            choices = np.random.randint(0, selected_faces.shape[1], size=len(edge_indices))
            sharp_face_indices = selected_faces[np.arange(len(edge_indices)), choices].astype(np.int32)

        face_indices = np.concatenate([sharp_face_indices, face_indices], axis=0)
        print(
            f"[particulate-timing] sample_points_total points={num_points} sharp={len(points_sharp)} "
            f"uniform={len(points_uniform)} elapsed_s={time.perf_counter() - started:.2f}",
            flush=True,
        )
        return points, normals, sharp_flag, face_indices

    return sample_points_cached


def _sample_sharp_points_cached(module: Any, sharp_edge_cache: dict[int, dict[str, Any]], mesh: Any, num_points: int) -> Any:
    np = module.np
    cache = _get_sharp_edge_cache(module, sharp_edge_cache, mesh)
    sharp_edges = cache["edges"]
    sharp_edge_faces = cache["faces"]
    sharp_edge_normals = cache["normals"]
    weights = cache["weights"]
    if len(sharp_edges) == 0:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.int32),
            sharp_edge_faces,
        )

    stage = time.perf_counter()
    edge_indices = np.random.choice(len(sharp_edges), size=max(1, int(num_points)), replace=True, p=weights)
    w = np.random.rand(max(1, int(num_points)), 1)
    vertices = mesh.vertices
    edge_a = sharp_edges[edge_indices, 0]
    edge_b = sharp_edges[edge_indices, 1]
    samples = w * vertices[edge_a] + (1.0 - w) * vertices[edge_b]
    normals = sharp_edge_normals[edge_indices]
    print(
        f"[particulate-timing] sharp_sample_points={num_points} sharp_edges={len(sharp_edges)} "
        f"elapsed_s={time.perf_counter() - stage:.2f}",
        flush=True,
    )
    return samples, normals, edge_indices.astype(np.int32), sharp_edge_faces


def _get_sharp_edge_cache(module: Any, sharp_edge_cache: dict[int, dict[str, Any]], mesh: Any) -> dict[str, Any]:
    cache_key = id(mesh)
    if cache_key in sharp_edge_cache:
        return sharp_edge_cache[cache_key]

    np = module.np
    started = time.perf_counter()
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    adjacency_edges = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)

    if len(adjacency) == 0 or len(adjacency_edges) == 0:
        cache = {
            "edges": np.zeros((0, 2), dtype=np.int32),
            "faces": np.zeros((0, 2), dtype=np.int32),
            "normals": np.zeros((0, 3), dtype=np.float64),
            "weights": np.zeros((0,), dtype=np.float64),
        }
        sharp_edge_cache[cache_key] = cache
        return cache

    n1 = normals[adjacency[:, 0]]
    n2 = normals[adjacency[:, 1]]
    dot = np.einsum("ij,ij->i", n1, n2)
    valid_normals = (np.linalg.norm(n1, axis=1) > 1e-8) & (np.linalg.norm(n2, axis=1) > 1e-8)
    sharp_mask = (np.cos(np.radians(150)) < dot) & (dot < np.cos(np.radians(30))) & valid_normals
    sharp_edges = adjacency_edges[sharp_mask].astype(np.int32)
    sharp_edge_faces = adjacency[sharp_mask].astype(np.int32)

    if len(sharp_edges) > 0:
        sharp_edge_normals = n1[sharp_mask] + n2[sharp_mask]
        normal_lengths = np.linalg.norm(sharp_edge_normals, axis=1, keepdims=True)
        sharp_edge_normals = np.divide(
            sharp_edge_normals,
            np.maximum(normal_lengths, 1e-8),
            out=np.zeros_like(sharp_edge_normals),
        )
        edge_vectors = mesh.vertices[sharp_edges[:, 1]] - mesh.vertices[sharp_edges[:, 0]]
        weights = np.linalg.norm(edge_vectors, axis=1).astype(np.float64)
        weights_sum = float(weights.sum())
        weights = weights / weights_sum if weights_sum > 0 else np.full(len(sharp_edges), 1.0 / len(sharp_edges))
    else:
        sharp_edge_normals = np.zeros((0, 3), dtype=np.float64)
        weights = np.zeros((0,), dtype=np.float64)

    cache = {
        "edges": sharp_edges,
        "faces": sharp_edge_faces,
        "normals": sharp_edge_normals,
        "weights": weights,
    }
    sharp_edge_cache[cache_key] = cache
    print(
        f"[particulate-timing] sharp_edge_cache faces={len(mesh.faces)} adjacency={len(adjacency)} "
        f"sharp_edges={len(sharp_edges)} elapsed_s={time.perf_counter() - started:.2f}",
        flush=True,
    )
    return cache


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
