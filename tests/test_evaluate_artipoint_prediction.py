from scripts.evaluate_artipoint_prediction import evaluate, undirected_axis_error_deg


def test_undirected_axis_error_ignores_sign() -> None:
    assert undirected_axis_error_deg([0, 1, 0], [0, -1, 0]) == 0.0


def test_prismatic_evaluation_omits_axis_line_error() -> None:
    prediction = {
        "joint_type": "prismatic",
        "axis": [0.0, -1.0, 0.0],
        "center": [0.0, 0.0, 0.0],
        "start_frame": 2,
        "end_frame": 6,
    }
    scene = {"drawer": {"axis": [0.0, 1.0, 0.0], "position": [1.0, 2.0, 3.0]}}

    metrics = evaluate(prediction, scene, "drawer", "prismatic")

    assert metrics["joint_type_correct"] is True
    assert metrics["axis_angle_error_deg_type_correct"] == 0.0
    assert metrics["axis_line_error"] is None
    assert metrics["axis_line_error_reason"] == "not_defined_for_prismatic"
