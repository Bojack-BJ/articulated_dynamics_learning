from __future__ import annotations

import argparse
import importlib.util
import sys
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
    original_prepare_inputs = module.prepare_inputs

    def prepare_inputs_with_global_points(*args: Any, **kwargs: Any) -> Any:
        kwargs["num_points_global"] = max(1, int(known.num_points_global))
        return original_prepare_inputs(*args, **kwargs)

    module.prepare_inputs = prepare_inputs_with_global_points
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


if __name__ == "__main__":
    raise SystemExit(main())
