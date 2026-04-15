from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PartSegmentationPart:
    part_id: int
    name: str
    role: str
    body_id: int
    body_name: str
    parent_body_id: int | None
    parent_body_name: str | None
    geom_ids: list[int] = field(default_factory=list)
    geom_names: list[str] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    visible_geom_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_mujoco_body_part_segmentation(
    mujoco,
    model,
    root_body_id: int,
    target_geom_ids: set[int],
    hidden_geom_ids: set[int] | None = None,
) -> dict[str, Any]:
    hidden_geom_ids = hidden_geom_ids or set()
    descendant_body_ids = _collect_descendant_body_ids(model, int(root_body_id))
    target_geom_ids = {int(geom_id) for geom_id in target_geom_ids}

    parts: list[PartSegmentationPart] = []
    part_id = 1
    for body_id in sorted(descendant_body_ids):
        body_geom_ids = sorted(
            geom_id for geom_id in target_geom_ids if int(model.geom_bodyid[geom_id]) == body_id
        )
        if not body_geom_ids:
            continue

        body_name = _body_name(mujoco, model, body_id)
        parent_body_id = int(model.body_parentid[body_id])
        parent_part_body_id = parent_body_id if parent_body_id in descendant_body_ids else None
        joint_names = [
            _joint_name(mujoco, model, joint_id)
            for joint_id in range(model.njnt)
            if int(model.jnt_bodyid[joint_id]) == body_id
        ]
        visible_geom_ids = [geom_id for geom_id in body_geom_ids if geom_id not in hidden_geom_ids]
        role = "base" if body_id == int(root_body_id) else ("articulated" if joint_names else "fixed_child")
        parts.append(
            PartSegmentationPart(
                part_id=part_id,
                name=body_name,
                role=role,
                body_id=body_id,
                body_name=body_name,
                parent_body_id=parent_part_body_id,
                parent_body_name=(
                    _body_name(mujoco, model, parent_part_body_id)
                    if parent_part_body_id is not None
                    else None
                ),
                geom_ids=body_geom_ids,
                geom_names=[_geom_name(mujoco, model, geom_id) for geom_id in body_geom_ids],
                joint_names=joint_names,
                visible_geom_ids=visible_geom_ids,
                metadata={
                    "geom_count": len(body_geom_ids),
                    "visible_geom_count": len(visible_geom_ids),
                },
            )
        )
        part_id += 1

    return {
        "provider": "mujoco-body-geom-prior",
        "version": 1,
        "mask_encoding": "indexed-mask-u16",
        "background_part_id": 0,
        "parts": [part.to_dict() for part in parts],
    }


def geom_to_part_id_lookup(part_segmentation: dict[str, Any] | None) -> dict[int, int]:
    if not isinstance(part_segmentation, dict):
        return {}
    lookup: dict[int, int] = {}
    for raw_part in part_segmentation.get("parts", []):
        if not isinstance(raw_part, dict):
            continue
        part_id = int(raw_part.get("part_id", 0))
        for geom_id in raw_part.get("geom_ids", []):
            lookup[int(geom_id)] = part_id
    return lookup


def part_name_lookup(part_segmentation: dict[str, Any] | None) -> dict[int, str]:
    if not isinstance(part_segmentation, dict):
        return {}
    return {
        int(raw_part.get("part_id", 0)): str(raw_part.get("name", f"part_{raw_part.get('part_id', 0)}"))
        for raw_part in part_segmentation.get("parts", [])
        if isinstance(raw_part, dict) and int(raw_part.get("part_id", 0)) > 0
    }


def segmentation_to_part_mask_u16(
    mujoco,
    segmentation,
    target_geom_ids: set[int],
    part_segmentation: dict[str, Any] | None,
):
    import numpy as np

    if segmentation.ndim != 3 or segmentation.shape[2] < 2:
        raise ValueError("Expected MuJoCo segmentation render with at least 2 channels")

    geom_ids = segmentation[:, :, 0].astype(np.int32, copy=False)
    object_types = segmentation[:, :, 1].astype(np.int32, copy=False)
    geom_type_id = int(mujoco.mjtObj.mjOBJ_GEOM)
    target_geom_ids_np = np.array(sorted(int(geom_id) for geom_id in target_geom_ids), dtype=np.int32)
    mask = np.isin(geom_ids, target_geom_ids_np)
    mask &= object_types == geom_type_id

    output = np.zeros(geom_ids.shape, dtype=np.uint16)
    geom_to_part = geom_to_part_id_lookup(part_segmentation)
    if not geom_to_part:
        output[mask] = np.uint16(1)
        return output

    for geom_id, part_id in geom_to_part.items():
        output[mask & (geom_ids == int(geom_id))] = np.uint16(max(0, int(part_id)))
    return output


def _collect_descendant_body_ids(model, root_body_id: int) -> set[int]:
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


def _body_name(mujoco, model, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name if name else f"body_{body_id}"


def _geom_name(mujoco, model, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name else f"geom_{geom_id}"


def _joint_name(mujoco, model, joint_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    return name if name else f"joint_{joint_id}"
