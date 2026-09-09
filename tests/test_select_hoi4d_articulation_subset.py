from scripts.select_hoi4d_articulation_subset import select_sequences


def test_core_profile_keeps_only_large_articulation_tasks(tmp_path):
    release = tmp_path / "release.txt"
    release.write_text("cam/H1/C3/N1/S1/s1/T1\ncam/H1/C3/N1/S1/s1/T2\ncam/H1/C4/N1/S1/s1/T2\n")
    rows = select_sequences(release, "core")
    assert [row["sequence"] for row in rows] == [
        "cam/H1/C3/N1/S1/s1/T2",
        "cam/H1/C4/N1/S1/s1/T2",
    ]
