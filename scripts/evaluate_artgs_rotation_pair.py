#!/usr/bin/env python3
"""Compare a completed full ArtGS rerun with canonical saved predictions."""
import argparse
import sys
import numpy as np
from evaluate_external_axis_common import read, write, ground_truth, load_prediction, part_mapping, joint_metrics


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "src"))
    status = read(args.work/"status.json")
    if status["status"] != "complete":
        raise RuntimeError("Native optimization/export has not completed")
    obj = args.root/"outputs/external_baseline_suite_v1/per_object"/status["object"]
    ref, labels, bbox, gt = ground_truth(args.root,status["object"])
    q = np.asarray(status["rotation"])
    result = {"object":status["object"],"rotation":q.tolist(),"protocol":"full native rerun; child part IoU matching"}
    for name, adapter in [("canonical",None),("rotated",args.work/"adapter")]:
        points, pred_labels, joints = load_prediction(obj,"artgs",adapter)
        if name == "rotated":
            points = points @ q
            for joint in joints:
                joint["axis"] = (q.T @ joint["axis"]).tolist()
                joint["origin"] = (q.T @ joint["origin"]).tolist()
        mapping, overlap = part_mapping(points,pred_labels,ref,labels,bbox)
        result[name] = {"mapping":mapping,"overlap":overlap,"joints":joint_metrics(joints,gt,mapping,bbox)}
    write(args.work/"paired_metrics.json",result)
    print(result)


if __name__ == "__main__":
    main()
