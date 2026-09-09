import numpy as np

from scripts.export_arti4d_to_track2art import official_optical_camera_pose


def test_official_optical_pose_applies_image_flip() -> None:
    angle = np.deg2rad(35.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    optical_pose = np.eye(4)
    optical_pose[:3, :3] = rotation
    optical_pose[:3, 3] = [0.4, -0.2, 1.3]

    optical_point = np.array([0.2, 0.3, 1.1, 1.0])
    flipped_point = np.array([-0.2, -0.3, 1.1, 1.0])
    adapted_pose = official_optical_camera_pose(optical_pose)

    np.testing.assert_allclose(
        adapted_pose @ optical_point,
        optical_pose @ flipped_point,
        atol=1e-9,
    )
    np.testing.assert_allclose(np.linalg.det(adapted_pose[:3, :3]), 1.0)
