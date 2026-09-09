import json
import subprocess
import sys
from pathlib import Path


def test_applies_track_override_to_visualization_artifact(tmp_path: Path) -> None:
    slots = tmp_path / "slots.json"
    override = tmp_path / "override.json"
    output = tmp_path / "oracle.json"
    slots.write_text(json.dumps({"tracks": [
        {"track_id": 10, "part_id": 3}, {"track_id": 11, "part_id": 4}
    ]}))
    override.write_text(json.dumps({"track_to_slot": {"10": 7, "11": 13}}))

    subprocess.run([
        sys.executable,
        "scripts/apply_oracle_slot_override.py",
        str(slots), str(override), str(output),
    ], check=True)

    result = json.loads(output.read_text())
    assert [track["part_id"] for track in result["tracks"]] == [7, 13]
    assert result["part_track_counts"] == {"7": 1, "13": 1}
