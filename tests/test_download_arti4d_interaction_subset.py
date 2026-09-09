from scripts.download_arti4d_interaction_subset import archive_scene_prefix


def test_archive_scene_prefix_finds_nested_scene() -> None:
    names = [
        "rh078/scene_a/rgb/000001.jpg",
        "rh078/scene_a/depth/000001.png",
        "rh078/scene_b/rgb/000001.jpg",
    ]

    assert archive_scene_prefix(names, "scene_a") == "rh078/scene_a/"


def test_archive_scene_prefix_rejects_missing_scene() -> None:
    try:
        archive_scene_prefix(["rh078/scene_a/rgb/000001.jpg"], "scene_b")
    except ValueError as exc:
        assert "scene_b" in str(exc)
    else:
        raise AssertionError("missing scenes must be rejected")
