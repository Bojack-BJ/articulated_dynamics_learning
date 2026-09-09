from __future__ import annotations

import csv
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET

from .gapartnet_adapter import GAPartNetMJCFAdapter, GAPartNetMJCFConfig


DEFAULT_PILOT_IDS = ("7304", "12042", "12531", "45267", "45855")
SUPPORTED_JOINT_TYPES = {"revolute", "continuous", "prismatic"}


@dataclass(frozen=True)
class GAPartNetDatasetConfig:
    archive_path: Path
    output_dir: Path
    object_ids: tuple[str, ...] = DEFAULT_PILOT_IDS
    force: bool = False
    density_kg_m3: float = 30.0


def _project_category(model_category: str, joint_type: str) -> str:
    normalized = model_category.strip().lower()
    if normalized == "storagefurniture":
        return "drawer" if joint_type == "prismatic" else "door"
    aliases = {
        "coffeemachine": "coffeemachine",
        "dishwasher": "dishwasher",
        "microwave": "microwave",
        "refrigerator": "refrigerator",
        "oven": "oven",
        "door": "door",
    }
    if normalized in aliases:
        return aliases[normalized]
    return "drawer" if joint_type == "prismatic" else "door"


def _movable_joints(urdf_root: ET.Element) -> list[ET.Element]:
    return [
        joint
        for joint in urdf_root.findall("joint")
        if str(joint.get("type", "fixed")).lower() in SUPPORTED_JOINT_TYPES
    ]


def _safe_extract_prefix(archive: zipfile.ZipFile, prefix: str, destination: Path) -> None:
    prefix_path = PurePosixPath(prefix)
    members = [name for name in archive.namelist() if PurePosixPath(name).is_relative_to(prefix_path)]
    if not members:
        raise FileNotFoundError(f"Object prefix not found in GAPartNet archive: {prefix}")
    for member in members:
        relative = PurePosixPath(member).relative_to(prefix_path)
        if not relative.parts:
            continue
        output = destination.joinpath(*relative.parts)
        if member.endswith("/"):
            output.mkdir(parents=True, exist_ok=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target)


class GAPartNetDatasetPreparer:
    def prepare(self, config: GAPartNetDatasetConfig) -> dict[str, Any]:
        archive_path = Path(config.archive_path).expanduser().resolve()
        output_dir = Path(config.output_dir).expanduser().resolve()
        assets_dir = output_dir / "assets"
        models_dir = output_dir / "models"
        assets_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        with zipfile.ZipFile(archive_path) as archive:
            for object_id in config.object_ids:
                prefix = f"partnet_mobility_part/{object_id}/"
                asset_dir = assets_dir / object_id
                if config.force and asset_dir.exists():
                    shutil.rmtree(asset_dir)
                if not asset_dir.exists():
                    _safe_extract_prefix(archive, prefix, asset_dir)

                meta = json.loads((asset_dir / "meta.json").read_text(encoding="utf-8"))
                urdf_path = asset_dir / "mobility_annotation_gapartnet.urdf"
                urdf_root = ET.parse(urdf_path).getroot()
                movable = _movable_joints(urdf_root)
                if len(movable) != 1:
                    raise ValueError(
                        f"Pilot asset {object_id} must have exactly one movable joint; found {len(movable)}"
                    )
                joint = movable[0]
                joint_type = str(joint.get("type")).lower()
                project_category = _project_category(str(meta.get("model_cat", "unknown")), joint_type)
                model_path = models_dir / f"gapartnet_{object_id}.xml"
                GAPartNetMJCFAdapter().convert(
                    GAPartNetMJCFConfig(
                        asset_dir=asset_dir,
                        output_mjcf=model_path,
                        density_kg_m3=config.density_kg_m3,
                    )
                )
                bbox = json.loads((asset_dir / "bounding_box.json").read_text(encoding="utf-8"))
                lower = [float(value) for value in bbox["min"]]
                upper = [float(value) for value in bbox["max"]]
                extent = [upper[index] - lower[index] for index in range(3)]
                center = [(upper[index] + lower[index]) * 0.5 for index in range(3)]
                rows.append(
                    {
                        "source_object_id": object_id,
                        "object_id": f"gapartnet_{project_category}_{object_id}",
                        "source_category": str(meta.get("model_cat", "unknown")),
                        "category": project_category,
                        "joint_name": str(joint.get("name")),
                        "joint_type": joint_type,
                        "asset_dir": str(asset_dir),
                        "model_path": str(model_path),
                        "bbox_center": center,
                        "bbox_extent": extent,
                        "recommended_camera_distance": max(extent) * 1.8,
                    }
                )

        catalog_path = output_dir / "catalog.json"
        catalog_path.write_text(json.dumps({"archive": str(archive_path), "objects": rows}, indent=2) + "\n")
        manifest_path = output_dir / "recording_batch.tsv"
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["# category", "model_path", "object_id", "joint_name"])
            for row in rows:
                writer.writerow([row["category"], row["model_path"], row["object_id"], row["joint_name"]])
        return {
            "archive_path": str(archive_path),
            "output_dir": str(output_dir),
            "catalog_path": str(catalog_path),
            "batch_manifest_path": str(manifest_path),
            "object_count": len(rows),
            "objects": rows,
        }
