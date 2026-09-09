#!/usr/bin/env python3
"""Extract and convert representative PartNet-Mobility articulation profiles."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree as ET

from rgbd_urdf_mvp.sim.gapartnet_adapter import GAPartNetMJCFAdapter, GAPartNetMJCFConfig
from rgbd_urdf_mvp.sim.gapartnet_dataset import _movable_joints, _project_category


DEFAULT_SAMPLES = {
    "single_revolute": "12531",
    "single_prismatic": "45855",
    # Two similarly sized moving panels provide substantially better RGB-D
    # observability than Door 8867, whose second hinge controls a tiny detail.
    "multi_revolute": "8961",
    "multi_prismatic": "45746",
    "mixed": "40147",
}


def _extract_object(
    archive: zipfile.ZipFile,
    object_id: str,
    destination: Path,
    members: list[str] | None = None,
) -> None:
    prefix = PurePosixPath(f"dataset/{object_id}")
    if members is None:
        members = [name for name in archive.namelist() if PurePosixPath(name).is_relative_to(prefix)]
    if not members:
        raise FileNotFoundError(f"PartNet-Mobility object not found: {object_id}")
    for member in members:
        relative = PurePosixPath(member).relative_to(prefix)
        if not relative.parts:
            continue
        output = destination.joinpath(*relative.parts)
        if member.endswith("/"):
            output.mkdir(parents=True, exist_ok=True)
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, output.open("wb") as target:
            shutil.copyfileobj(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", action="append", help="Override as PROFILE=OBJECT_ID")
    parser.add_argument(
        "--split-catalog",
        type=Path,
        default=None,
        help="Prepare every object from a catalog.tsv produced by build_partnet_mobility_splits.py",
    )
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    samples: list[dict[str, str]] = []
    if args.split_catalog is not None:
        with args.split_catalog.expanduser().resolve().open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                samples.append(
                    {
                        "profile": str(row["articulation_profile"]),
                        "object_id": str(row["object_id"]),
                        "split": str(row["split"]),
                    }
                )
        if args.max_objects is not None:
            samples = samples[: max(0, int(args.max_objects))]
    else:
        samples = [
            {"profile": profile, "object_id": object_id, "split": "sample"}
            for profile, object_id in DEFAULT_SAMPLES.items()
        ]
    for raw in args.sample or []:
        profile, separator, object_id = raw.partition("=")
        if not separator or not profile or not object_id:
            parser.error("--sample must use PROFILE=OBJECT_ID")
        if args.split_catalog is None:
            samples = [sample for sample in samples if sample["profile"] != profile]
        samples.append({"profile": profile, "object_id": object_id, "split": "sample"})

    output_dir = args.output_dir.expanduser().resolve()
    assets_dir = output_dir / "assets"
    models_dir = output_dir / "models"
    assets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    failures: list[dict[str, str]] = []
    with zipfile.ZipFile(args.archive.expanduser().resolve()) as archive:
        requested_ids = {sample["object_id"] for sample in samples}
        indexed_members = {object_id: [] for object_id in requested_ids}
        for member in archive.namelist():
            fields = PurePosixPath(member).parts
            if len(fields) >= 2 and fields[0] == "dataset" and fields[1] in indexed_members:
                indexed_members[fields[1]].append(member)
        for sample in samples:
            profile = sample["profile"]
            object_id = sample["object_id"]
            asset_dir = assets_dir / object_id
            if args.force and asset_dir.exists():
                shutil.rmtree(asset_dir)
            if not asset_dir.exists():
                _extract_object(archive, object_id, asset_dir, indexed_members[object_id])
            try:
                meta = json.loads((asset_dir / "meta.json").read_text(encoding="utf-8"))
                urdf = ET.parse(asset_dir / "mobility.urdf").getroot()
                movable = _movable_joints(urdf)
                joint_types = [str(joint.get("type", "fixed")).lower() for joint in movable]
                primary_type = "prismatic" if joint_types and all(kind == "prismatic" for kind in joint_types) else "revolute"
                category = _project_category(str(meta.get("model_cat", "unknown")), primary_type)
                model_path = models_dir / f"partnet_{object_id}.xml"
                if args.force or not model_path.exists():
                    GAPartNetMJCFAdapter().convert(
                        GAPartNetMJCFConfig(
                            asset_dir=asset_dir,
                            output_mjcf=model_path,
                            urdf_name="mobility.urdf",
                        )
                    )
            except Exception as error:
                if not args.continue_on_error:
                    raise
                failures.append(
                    {
                        "object_id": object_id,
                        "profile": profile,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            rows.append(
                {
                    "profile": profile,
                    "source_object_id": object_id,
                    "source_category": str(meta.get("model_cat", "unknown")),
                    "category": category,
                    "object_id": (
                        f"partnet_{object_id}"
                        if args.split_catalog is not None
                        else f"partnet_{profile}_{object_id}"
                    ),
                    "split": sample["split"],
                    "model_path": str(model_path),
                    "joint_name": "auto",
                    "joint_count": len(movable),
                    "joint_types": joint_types,
                }
            )

    (output_dir / "catalog.json").write_text(json.dumps({"objects": rows}, indent=2) + "\n", encoding="utf-8")
    (output_dir / "failures.json").write_text(
        json.dumps({"failure_count": len(failures), "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "recording_batch.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["# category", "model_path", "object_id", "joint_name"])
        for row in rows:
            writer.writerow([row["category"], row["model_path"], row["object_id"], row["joint_name"]])
    print(
        json.dumps(
            {"output_dir": str(output_dir), "object_count": len(rows), "failure_count": len(failures)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
