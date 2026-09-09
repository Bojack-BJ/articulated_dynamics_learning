import csv
import subprocess
import sys
from pathlib import Path


def test_build_manifest_repeats_train_and_keeps_validation_once(tmp_path: Path) -> None:
    base = tmp_path / "base.tsv"
    base.write_text("object_id\ttracks_path\tfeatures_npz\tsplit\npartnet_1\ta\tb\ttrain\n")
    root = tmp_path / "arti4d"
    for name in ("train_obj", "val_obj"):
        scene = root / name
        (scene / "track_quality").mkdir(parents=True)
        (scene / "tracking").mkdir()
        (scene / "track_quality/motion_part_tracks_with_quality.json").write_text("{}")
        (scene / "tracking/cotracker_features.npz").write_bytes(b"npz")
        (scene / "relation_gt.json").write_text("{}")
    output = tmp_path / "merged.tsv"
    subprocess.run([
        sys.executable, "scripts/build_arti4d_relation_manifest.py", str(base), str(root),
        str(output), "--train", "train_obj", "--val", "val_obj", "--repeat", "3",
    ], check=True)
    with output.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 5
    assert sum(row["split"] == "train" for row in rows) == 4
    assert sum(row["split"] == "val" for row in rows) == 1
    assert rows[-1]["relation_gt_path"].endswith("val_obj/relation_gt.json")
