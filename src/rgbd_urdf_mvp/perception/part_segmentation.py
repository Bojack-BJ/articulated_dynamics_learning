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
    """Build semantic parts as maximal fixed-connected MuJoCo body groups.

    A body carrying a joint starts a new kinematic part. Bodies without a joint
    are rigidly attached to their parent and therefore inherit its part. The
    root body always anchors the base part even when it has a free placement
    joint. Raw body membership is retained for diagnostics.
    """
    hidden_geom_ids = hidden_geom_ids or set()
    descendant_body_ids = _collect_descendant_body_ids(model, int(root_body_id))
    target_geom_ids = {int(geom_id) for geom_id in target_geom_ids}

    body_joint_names = {
        body_id: [
            _joint_name(mujoco, model, joint_id)
            for joint_id in range(model.njnt)
            if int(model.jnt_bodyid[joint_id]) == body_id
        ]
        for body_id in descendant_body_ids
    }
    body_anchor: dict[int, int] = {}
    for body_id in _body_tree_order(model, int(root_body_id), descendant_body_ids):
        if body_id == int(root_body_id) or body_joint_names[body_id]:
            body_anchor[body_id] = body_id
        else:
            parent_body_id = int(model.body_parentid[body_id])
            body_anchor[body_id] = body_anchor[parent_body_id]

    anchor_members: dict[int, list[int]] = {}
    for body_id, anchor_body_id in body_anchor.items():
        anchor_members.setdefault(anchor_body_id, []).append(body_id)

    parts: list[PartSegmentationPart] = []
    body_to_part_id: dict[int, int] = {}
    for body_id in sorted(anchor_members):
        member_body_ids = sorted(anchor_members[body_id])
        body_geom_ids = sorted(
            geom_id
            for geom_id in target_geom_ids
            if int(model.geom_bodyid[geom_id]) in member_body_ids
        )
        if not body_geom_ids:
            continue
        part_id = len(parts) + 1

        body_name = _body_name(mujoco, model, body_id)
        parent_body_id = int(model.body_parentid[body_id])
        parent_part_body_id = (
            body_anchor[parent_body_id]
            if parent_body_id in descendant_body_ids and body_id != int(root_body_id)
            else None
        )
        joint_names = body_joint_names[body_id]
        visible_geom_ids = [geom_id for geom_id in body_geom_ids if geom_id not in hidden_geom_ids]
        role = "base" if body_id == int(root_body_id) else "articulated"
        for member_body_id in member_body_ids:
            body_to_part_id[member_body_id] = part_id
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
                    "member_body_ids": member_body_ids,
                    "member_body_names": [
                        _body_name(mujoco, model, member_body_id)
                        for member_body_id in member_body_ids
                    ],
                    "fixed_connected_body_count": len(member_body_ids),
                },
            )
        )

    return {
        "provider": "mujoco-body-geom-prior",
        "version": 2,
        "ontology": "maximal-fixed-joint-connected-components",
        "mask_encoding": "indexed-mask-u16",
        "background_part_id": 0,
        "parts": [part.to_dict() for part in parts],
        "raw_body_to_part_id": {
            str(body_id): int(part_id) for body_id, part_id in sorted(body_to_part_id.items())
        },
    }


def collapse_fixed_connected_parts(part_segmentation: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy one-body-per-part artifact to the kinematic ontology.

    Legacy recorder artifacts identify a fixed child through ``role=fixed_child``
    and its ``parent_body_id``. This conversion is deterministic and idempotent;
    it never uses visibility, track coverage, or predictions from a method.
    """
    if part_segmentation.get("ontology") == "maximal-fixed-joint-connected-components":
        return part_segmentation
    raw_parts = [part for part in part_segmentation.get("parts", []) if isinstance(part, dict)]
    by_body = {int(part["body_id"]): part for part in raw_parts if part.get("body_id") is not None}
    anchor_by_body: dict[int, int] = {}

    def anchor(body_id: int) -> int:
        if body_id in anchor_by_body:
            return anchor_by_body[body_id]
        part = by_body[body_id]
        parent = part.get("parent_body_id")
        if part.get("role") != "fixed_child" or parent is None or int(parent) not in by_body:
            value = body_id
        else:
            value = anchor(int(parent))
        anchor_by_body[body_id] = value
        return value

    for body_id in by_body:
        anchor(body_id)
    groups: dict[int, list[dict[str, Any]]] = {}
    for body_id, part in by_body.items():
        groups.setdefault(anchor_by_body[body_id], []).append(part)

    collapsed_parts: list[dict[str, Any]] = []
    raw_part_to_part_id: dict[str, int] = {}
    for new_part_id, anchor_body_id in enumerate(sorted(groups), start=1):
        members = sorted(groups[anchor_body_id], key=lambda item: int(item["body_id"]))
        anchor_part = by_body[anchor_body_id]
        collapsed = dict(anchor_part)
        collapsed["part_id"] = new_part_id
        collapsed["role"] = "base" if anchor_part.get("role") == "base" else "articulated"
        collapsed["geom_ids"] = sorted(
            {int(value) for member in members for value in member.get("geom_ids", [])}
        )
        collapsed["geom_names"] = [
            str(value) for member in members for value in member.get("geom_names", [])
        ]
        collapsed["visible_geom_ids"] = sorted(
            {int(value) for member in members for value in member.get("visible_geom_ids", [])}
        )
        metadata = dict(collapsed.get("metadata", {}))
        metadata.update({
            "member_body_ids": [int(member["body_id"]) for member in members],
            "member_body_names": [str(member.get("body_name", "")) for member in members],
            "fixed_connected_body_count": len(members),
            "legacy_raw_part_ids": [int(member["part_id"]) for member in members],
        })
        collapsed["metadata"] = metadata
        collapsed_parts.append(collapsed)
        for member in members:
            raw_part_to_part_id[str(int(member["part_id"]))] = new_part_id

    output = dict(part_segmentation)
    output.update({
        "version": 2,
        "ontology": "maximal-fixed-joint-connected-components",
        "parts": collapsed_parts,
        "raw_part_to_part_id": raw_part_to_part_id,
        "legacy_raw_parts": raw_parts,
    })
    return output


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


def _body_tree_order(model, root_body_id: int, body_ids: set[int]) -> list[int]:
    ordered: list[int] = []
    pending = [int(root_body_id)]
    while pending:
        body_id = pending.pop(0)
        ordered.append(body_id)
        pending.extend(
            sorted(
                candidate
                for candidate in body_ids
                if candidate not in ordered
                and candidate not in pending
                and int(model.body_parentid[candidate]) == body_id
            )
        )
    return ordered


def _body_name(mujoco, model, body_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    return name if name else f"body_{body_id}"


def _geom_name(mujoco, model, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name else f"geom_{geom_id}"


def _joint_name(mujoco, model, joint_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    return name if name else f"joint_{joint_id}"
