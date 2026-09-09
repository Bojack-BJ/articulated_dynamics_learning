from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts" / "record_aligned_aim_style.py"
SPEC = importlib.util.spec_from_file_location("record_aligned_aim", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_aim_style_recorder_exposes_cli() -> None:
    assert callable(MODULE.parse_args)
    assert callable(MODULE.main)


def test_build_command_uses_per_object_frame_schedule(tmp_path: Path) -> None:
    args = SimpleNamespace(
        python=Path("/python"),
        output_root=tmp_path / "recordings",
        mujoco_gl=None,
    )
    command = MODULE._build_command(
        args,
        {
            "object_id": "partnet_47648",
            "category": "door",
            "interaction_frames": "500",
            "fps": "15",
        },
        tmp_path / "partnet_47648.xml",
    )
    assert command[command.index("--duration-s") + 1] == "33.3333333333"
    assert command[command.index("--fps") + 1] == "15"
    assert command[command.index("--recording-protocol") + 1] == "aim_style"


def test_build_command_keeps_aligned_defaults(tmp_path: Path) -> None:
    args = SimpleNamespace(
        python=Path("/python"),
        output_root=tmp_path / "recordings",
        mujoco_gl=None,
    )
    command = MODULE._build_command(
        args,
        {"object_id": "partnet_1", "category": "drawer"},
        tmp_path / "partnet_1.xml",
    )
    assert command[command.index("--duration-s") + 1] == "8"
    assert command[command.index("--fps") + 1] == "15"


def test_build_command_supports_fixed_end_protocol(tmp_path: Path) -> None:
    args = SimpleNamespace(
        python=Path("/python"),
        output_root=tmp_path / "recordings",
        mujoco_gl=None,
    )
    command = MODULE._build_command(
        args,
        {
            "object_id": "partnet_1",
            "category": "door",
            "recording_protocol": "aim_style_fixed_end",
            "aim_static_scan_views": "100",
            "aim_end_scan_views": "16",
        },
        tmp_path / "partnet_1.xml",
    )
    assert command[command.index("--recording-protocol") + 1] == "aim_style_fixed_end"
    assert command[command.index("--aim-static-scan-views") + 1] == "100"
    assert command[command.index("--aim-end-scan-views") + 1] == "16"
