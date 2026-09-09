from scripts.build_oracle_slot_override import build_override


def test_build_override_matches_parts_to_distinct_max_overlap_slots() -> None:
    tracks = {"tracks": [
        {"track_id": 1, "part_id": 1},
        {"track_id": 2, "part_id": 1},
        {"track_id": 3, "part_id": 2},
        {"track_id": 4, "part_id": 2},
    ]}
    prediction = {
        "track_ids": [1, 2, 3, 4],
        "track_slot_assignments": [7, 7, 1, 1],
    }
    result = build_override(tracks, prediction)
    assert result["part_to_slot"] == {"1": 7, "2": 1}
    assert set(result["track_to_slot"].values()) == {1, 7}
