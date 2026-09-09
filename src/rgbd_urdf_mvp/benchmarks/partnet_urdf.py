"""Extract kinematic metadata from PartNet-Mobility URDF assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


MOVABLE_JOINT_TYPES = frozenset({"continuous", "revolute", "prismatic"})


@dataclass(frozen=True)
class PartNetJointMetadata:
    """Kinematic inventory read directly from one PartNet-Mobility URDF."""

    joint_count: int
    joint_types: tuple[str, ...]
    joint_names: tuple[str, ...]


def load_partnet_joint_metadata(urdf_path: str | Path) -> PartNetJointMetadata:
    """Return movable joints, excluding fixed and unsupported free joints."""
    root = ET.parse(Path(urdf_path)).getroot()
    joints: list[tuple[str, str]] = []
    for joint in root.findall(".//joint"):
        joint_type = str(joint.get("type", "")).strip().lower()
        if joint_type not in MOVABLE_JOINT_TYPES:
            continue
        joints.append((str(joint.get("name", "")), joint_type))
    return PartNetJointMetadata(
        joint_count=len(joints),
        joint_types=tuple(joint_type for _, joint_type in joints),
        joint_names=tuple(name for name, _ in joints),
    )


def resolve_partnet_urdf(
    assets_root: str | Path, object_id: str
) -> Path | None:
    """Resolve common PartNet asset layouts for an aligned object id."""
    root = Path(assets_root)
    numeric_id = object_id.removeprefix("partnet_")
    candidates = (
        root / numeric_id / "mobility.urdf",
        root / object_id / "mobility.urdf",
        root / numeric_id / "mobility_v2.urdf",
        root / object_id / "mobility_v2.urdf",
    )
    return next((path for path in candidates if path.is_file()), None)
