import json

import numpy as np
from PIL import Image

from scripts.finalize_curated_object_masks import finalize_episode


def test_finalize_episode_merges_nonzero_part_labels(tmp_path):
    indexed = np.array([[0, 1], [2, 0]], dtype=np.uint16)
    Image.fromarray(indexed, mode="I;16").save(tmp_path / "indexed.png")
    source = tmp_path / "episode.annotation.json"
    source.write_text(json.dumps({
        "frames": [{"part_mask_path": "indexed.png"}],
        "metadata": {"uses_gt_part_masks": True, "part_segmentation": {"provider": "manual"}},
    }))

    output = tmp_path / "episode.curated.json"
    episode = finalize_episode(source, output, tmp_path / "binary")

    mask = np.asarray(Image.open(episode["frames"][0]["mask_path"]))
    assert mask.tolist() == [[0, 255], [255, 0]]
    assert episode["metadata"]["uses_gt_part_masks"] is False
    assert "part_segmentation" not in episode["metadata"]
