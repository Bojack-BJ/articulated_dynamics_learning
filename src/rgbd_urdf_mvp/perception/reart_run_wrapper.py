from __future__ import annotations

import argparse
import importlib.util
import json
import random
import runpy
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ReArt with a point-cloud-compatible real Sequence loader.")
    parser.add_argument("--reart-root", type=Path, required=True)
    parser.add_argument("--seq-path", type=Path, required=True)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--cano-idx", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-parts", type=int, default=10)
    parser.add_argument("--stage", choices=["base", "kinematic", "both", "evaluate"], default="base")
    parser.add_argument("--base-n-iter", type=int, default=2000)
    parser.add_argument("--kinematic-n-iter", type=int, default=200)
    parser.add_argument("--assign-iter", type=int, default=1000)
    parser.add_argument("--snapshot-gap", type=int, default=100)
    parser.add_argument("--use-assign-loss", action="store_true")
    parser.add_argument("--use-flow-loss", action="store_true")
    parser.add_argument("--use-nproc", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--base-result-path", type=Path, default=None)
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args(argv)

    reart_root = args.reart_root.expanduser().resolve()
    seq_path = args.seq_path.expanduser().resolve()
    save_root = args.save_root.expanduser().resolve()
    if not (reart_root / "run_real.py").exists():
        raise FileNotFoundError(f"Expected ReArt run_real.py under {reart_root}")
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing ReArt sequence path: {seq_path}")
    save_root.mkdir(parents=True, exist_ok=True)

    if str(reart_root) not in sys.path:
        sys.path.insert(0, str(reart_root))
    _install_optional_dependency_fallbacks()
    _install_visualization_fallbacks()
    _patch_reart_real_sequence(reart_root)

    timings: dict[str, float] = {}
    outputs: dict[str, Any] = {}
    started = time.perf_counter()
    if args.stage in {"base", "both"}:
        stage_started = time.perf_counter()
        _run_reart_real(
            reart_root,
            seq_path=seq_path,
            save_root=save_root,
            cano_idx=args.cano_idx,
            num_points=args.num_points,
            num_parts=args.num_parts,
            model="base",
            n_iter=args.base_n_iter,
            assign_iter=args.assign_iter,
            snapshot_gap=args.snapshot_gap,
            use_assign_loss=args.use_assign_loss,
            use_flow_loss=args.use_flow_loss,
            use_nproc=args.use_nproc,
            resume=args.resume,
            base_result_path=None,
            evaluate=args.stage == "evaluate",
            extra_args=args.extra_arg,
        )
        timings["base_s"] = time.perf_counter() - stage_started
        outputs["base_result_path"] = str(save_root / seq_path.name / "result.pkl")
        outputs["base_model_path"] = str(save_root / seq_path.name / "model.pth.tar")

    if args.stage in {"kinematic", "both", "evaluate"}:
        base_result_path = args.base_result_path
        if base_result_path is None and args.stage == "both":
            base_result_path = save_root / seq_path.name / "result.pkl"
        stage_started = time.perf_counter()
        _run_reart_real(
            reart_root,
            seq_path=seq_path,
            save_root=save_root,
            cano_idx=args.cano_idx,
            num_points=args.num_points,
            num_parts=args.num_parts,
            model="kinematic",
            n_iter=args.kinematic_n_iter,
            assign_iter=0,
            snapshot_gap=args.snapshot_gap,
            use_assign_loss=args.use_assign_loss,
            use_flow_loss=args.use_flow_loss,
            use_nproc=args.use_nproc,
            resume=args.resume if args.stage == "evaluate" else None,
            base_result_path=base_result_path,
            evaluate=args.stage == "evaluate",
            extra_args=args.extra_arg,
        )
        timings["kinematic_s"] = time.perf_counter() - stage_started
        outputs["kinematic_result_path"] = str(save_root / seq_path.name / "result.pkl")
        outputs["kinematic_model_path"] = str(save_root / seq_path.name / "model.pth.tar")

    timings["total_s"] = time.perf_counter() - started
    manifest = {
        "source": "rgbd_urdf_mvp_reart_wrapper",
        "reart_root": str(reart_root),
        "seq_path": str(seq_path),
        "save_root": str(save_root),
        "stage": args.stage,
        "num_points": int(args.num_points),
        "num_parts": int(args.num_parts),
        "cano_idx": int(args.cano_idx),
        "timings": timings,
        "outputs": outputs,
    }
    manifest_path = save_root / seq_path.name / "reart_result.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"reart_result": str(manifest_path), "timings": timings}, indent=2), flush=True)
    return 0


def _run_reart_real(
    reart_root: Path,
    *,
    seq_path: Path,
    save_root: Path,
    cano_idx: int,
    num_points: int,
    num_parts: int,
    model: str,
    n_iter: int,
    assign_iter: int,
    snapshot_gap: int,
    use_assign_loss: bool,
    use_flow_loss: bool,
    use_nproc: bool,
    resume: Path | None,
    base_result_path: Path | None,
    evaluate: bool,
    extra_args: list[str],
) -> None:
    argv = [
        "run_real.py",
        f"--seq_path={seq_path}",
        f"--save_root={save_root}",
        f"--cano_idx={int(cano_idx)}",
        f"--num_points={int(num_points)}",
        f"--num_parts={int(num_parts)}",
        f"--model={model}",
        f"--n_iter={int(n_iter)}",
        f"--assign_iter={int(assign_iter)}",
        f"--snapshot_gap={int(snapshot_gap)}",
    ]
    if use_assign_loss:
        argv.append("--use_assign_loss")
    if use_flow_loss:
        argv.append("--use_flow_loss")
    if use_nproc:
        argv.append("--use_nproc")
    if evaluate:
        argv.append("--evaluate")
    if resume is not None:
        argv.extend(["--resume", str(resume.expanduser().resolve())])
    if base_result_path is not None:
        argv.append(f"--base_result_path={base_result_path.expanduser().resolve()}")
    argv.extend(extra_args)
    old_argv = sys.argv
    old_cwd = Path.cwd()
    try:
        sys.argv = argv
        import os

        os.chdir(reart_root)
        runpy.run_path(str(reart_root / "run_real.py"), run_name="__main__")
    finally:
        import os

        os.chdir(old_cwd)
        sys.argv = old_argv


def _install_optional_dependency_fallbacks() -> None:
    """Provide slow torch fallbacks for ReArt CUDA extensions when they are absent."""
    if _module_spec_missing("chamferdist"):
        chamferdist_module = types.ModuleType("chamferdist")
        backend_module = types.ModuleType("chamferdist._C")

        def knn_points_idx(p1, p2, lengths1, lengths2, k, version):
            del lengths1, lengths2, version
            dists = _batched_squared_cdist(p1, p2)
            values, indices = torch.topk(dists, k=int(k), dim=-1, largest=False, sorted=False)
            return indices.contiguous(), values.contiguous()

        def knn_points_backward(p1, p2, lengths1, lengths2, idx, grad_dists):
            del lengths1, lengths2
            grad_p1 = torch.zeros_like(p1)
            grad_p2 = torch.zeros_like(p2)
            batch_count = p1.shape[0]
            for batch_index in range(batch_count):
                gather_idx = idx[batch_index].reshape(-1)
                p1_rep = p1[batch_index].unsqueeze(1).expand(-1, idx.shape[2], -1).reshape(-1, p1.shape[-1])
                p2_nn = p2[batch_index].index_select(0, gather_idx)
                grad = grad_dists[batch_index].reshape(-1, 1)
                delta = 2.0 * grad * (p1_rep - p2_nn)
                grad_p1[batch_index] += delta.reshape(p1.shape[1], idx.shape[2], p1.shape[-1]).sum(dim=1)
                grad_p2[batch_index].index_add_(0, gather_idx, -delta)
            return grad_p1, grad_p2

        backend_module.knn_points_idx = knn_points_idx
        backend_module.knn_points_backward = knn_points_backward
        chamferdist_module._C = backend_module
        sys.modules["chamferdist"] = chamferdist_module
        sys.modules["chamferdist._C"] = backend_module

    if _module_spec_missing("knn_cuda"):
        knn_module = types.ModuleType("knn_cuda")

        class KNN:
            def __init__(self, k: int = 1, transpose_mode: bool = False) -> None:
                self.k = int(k)
                self.transpose_mode = bool(transpose_mode)

            def __call__(self, ref, query):
                ref_points = _as_batched_points(ref)
                query_points = _as_batched_points(query)
                dists = _batched_squared_cdist(query_points, ref_points)
                values, indices = torch.topk(dists, k=self.k, dim=-1, largest=False, sorted=True)
                return values, indices

        knn_module.KNN = KNN
        sys.modules["knn_cuda"] = knn_module

    if _module_spec_missing("pointnet2_cuda"):
        pointnet2_module = types.ModuleType("networks.pointnet_lib.pointnet2_utils")
        pointnet2_module.furthest_point_sample = _furthest_point_sample_fallback
        pointnet2_module.ball_query = _ball_query_fallback
        sys.modules["networks.pointnet_lib.pointnet2_utils"] = pointnet2_module


def _install_visualization_fallbacks() -> None:
    viz_module = types.ModuleType("utils.viz_utils")

    def _write_placeholder(path: str | Path | None, label: str) -> None:
        if path is None:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{label} skipped by rgbd_urdf_mvp ReArt wrapper.\n", encoding="utf-8")

    def vis_pc(*args, **kwargs):
        _write_placeholder(kwargs.get("save_path"), "vis_pc")

    def vis_structure(*args, **kwargs):
        _write_placeholder(kwargs.get("save_path"), "vis_structure")

    def vis_pc_seq(*args, **kwargs):
        _write_placeholder(kwargs.get("save_path"), "vis_pc_seq")

    viz_module.vis_pc = vis_pc
    viz_module.vis_structure = vis_structure
    viz_module.vis_pc_seq = vis_pc_seq
    sys.modules["utils.viz_utils"] = viz_module


def _module_spec_missing(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is None
    except ModuleNotFoundError:
        return True


def _as_batched_points(points):
    if points.ndim != 3:
        raise ValueError(f"Expected batched points with rank 3, got shape {tuple(points.shape)}")
    if points.shape[-1] == 3:
        return points.contiguous()
    if points.shape[1] == 3:
        return points.transpose(1, 2).contiguous()
    return points.contiguous()


def _batched_squared_cdist(source, target):
    diff = source[:, :, None, :] - target[:, None, :, :]
    return (diff * diff).sum(dim=-1)


def _furthest_point_sample_fallback(xyz, npoint: int):
    batch_size, point_count, _channels = xyz.shape
    npoint = int(npoint)
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=xyz.device)
    distance = torch.full((batch_size, point_count), 1e10, dtype=xyz.dtype, device=xyz.device)
    farthest = torch.randint(0, point_count, (batch_size,), dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=xyz.device)
    for point_index in range(npoint):
        centroids[:, point_index] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = distance.max(dim=-1).indices
    return centroids


def _ball_query_fallback(radius: float, nsample: int, xyz, new_xyz):
    """Match ReArt's PointNet++ CPU ball-query semantics with torch tensors."""
    batch_size, point_count, _channels = xyz.shape
    sample_count = new_xyz.shape[1]
    nsample = min(int(nsample), point_count)
    distances = _batched_squared_cdist(new_xyz, xyz)
    nearest = distances.min(dim=-1).indices.unsqueeze(-1)
    indices = torch.arange(point_count, dtype=torch.long, device=xyz.device)
    indices = indices.view(1, 1, point_count).expand(batch_size, sample_count, point_count).clone()
    indices[distances > float(radius) ** 2] = point_count
    indices = indices.sort(dim=-1).values[:, :, :nsample]
    fallback = nearest.expand(batch_size, sample_count, nsample)
    return torch.where(indices == point_count, fallback, indices)


def _patch_reart_real_sequence(reart_root: Path) -> None:
    module_name = "dataset.dataset_real"
    dataset_path = reart_root / "dataset" / "dataset_real.py"
    spec = importlib.util.spec_from_file_location(module_name, dataset_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ReArt dataset loader: {dataset_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    class PointCloudSequence:
        def __init__(self, seq_dir: str, num_points: int = 4096, cano_idx: int = 0) -> None:
            self.num_points = int(num_points)
            self.cano_idx = int(cano_idx)
            self.seq_dir = seq_dir
            self.files = sorted(
                [path for path in Path(seq_dir).glob("*.*") if path.suffix.lower() in {".ply", ".obj", ".glb"}],
                key=_sequence_file_sort_key,
            )
            if not self.files:
                raise FileNotFoundError(f"No .ply/.obj/.glb frames under {seq_dir}")
            self.point_list = [_load_points_or_mesh_surface(path, self.num_points) for path in self.files]
            if not 0 <= self.cano_idx < len(self.point_list):
                raise ValueError(f"cano_idx {self.cano_idx} out of range for {len(self.point_list)} frames")
            cano_points = self.point_list[self.cano_idx]
            vmax, vmin = cano_points.max(axis=0), cano_points.min(axis=0)
            diag = vmax - vmin
            self.centroid = cano_points.mean(axis=0)
            norm = np.linalg.norm(diag)
            self.scale = np.array(1.0 / norm if norm > 1e-9 else 1.0)

        def __len__(self) -> int:
            return 1

        def __getitem__(self, item: int) -> dict[str, np.ndarray]:
            complete_pc_list = []
            for points in self.point_list:
                complete_pc_list.append(_sample_points(points, self.num_points))
            complete = np.stack(complete_pc_list).astype("float32")
            return {
                "cano_pc": complete[self.cano_idx],
                "pc_list": np.concatenate((complete[: self.cano_idx], complete[self.cano_idx + 1 :]), axis=0),
                "complete_pc_list": complete,
            }

    module.Sequence = PointCloudSequence


def _load_points_or_mesh_surface(path: Path, num_points: int) -> np.ndarray:
    if path.suffix.lower() == ".ply":
        points = _read_vertex_ply(path)
        if len(points) > 0:
            return points.astype("float32")
    import trimesh

    mesh = trimesh.load_mesh(path, process=False)
    if hasattr(mesh, "geometry"):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise ValueError(f"No geometry in scene: {path}")
        mesh = geometries[0]
    vertices = np.asarray(getattr(mesh, "vertices", []), dtype=np.float32)
    faces = np.asarray(getattr(mesh, "faces", []))
    if vertices.size and faces.size == 0:
        return vertices.astype("float32")
    pc, _face_idx = trimesh.sample.sample_surface(mesh, count=num_points)
    return pc.astype("float32")


def _read_vertex_ply(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines or lines[0].strip() != "ply":
        return np.empty((0, 3), dtype=np.float32)
    properties: list[str] = []
    vertex_count = 0
    data_start = None
    in_vertex = False
    for index, line in enumerate(lines[1:], start=1):
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "element":
            in_vertex = fields[1] == "vertex"
            if in_vertex:
                vertex_count = int(fields[2])
        elif fields[0] == "property" and in_vertex:
            properties.append(fields[-1])
        elif fields[0] == "end_header":
            data_start = index + 1
            break
    if data_start is None:
        return np.empty((0, 3), dtype=np.float32)
    prop = {name: idx for idx, name in enumerate(properties)}
    if not {"x", "y", "z"}.issubset(prop):
        return np.empty((0, 3), dtype=np.float32)
    points = []
    for line in lines[data_start : data_start + vertex_count]:
        values = line.split()
        if len(values) < len(properties):
            continue
        points.append([float(values[prop["x"]]), float(values[prop["y"]]), float(values[prop["z"]])])
    return np.asarray(points, dtype=np.float32)


def _sample_points(points: np.ndarray, count: int) -> np.ndarray:
    if len(points) == 0:
        raise ValueError("Cannot sample an empty point cloud.")
    if len(points) == count:
        return points.astype("float32")
    replace = len(points) < count
    indices = np.random.choice(len(points), size=count, replace=replace)
    return points[indices].astype("float32")


def _sequence_file_sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return (int(digits[-1]) if digits else 0, path.name)


if __name__ == "__main__":
    raise SystemExit(main())
