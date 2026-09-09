from scripts.build_artipoint_bidirectional_manifest import manifest_rows


def test_bidirectional_manifest_keeps_native_and_adapted_protocols_explicit() -> None:
    rows = manifest_rows()

    assert [row["dataset"] for row in rows] == ["partnet_mobility", "arti4d_rh078"]
    assert "artipoint_native" in rows[1]["methods"]
    assert "artipoint_object_mask_adapter" in rows[0]["methods"]
    assert "dta_unsupported_input" in rows[1]["methods"]
    assert rows[0]["gt_part_labels_in_inference"] is False
    assert rows[1]["gt_part_labels_in_inference"] is False
