#!/usr/bin/env python3
"""Add diagnostic-only sequential-RANSAC logging to an AiM checkout."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aim_root", type=Path)
    args = parser.parse_args()
    path = args.aim_root / "seg_main.py"
    text = path.read_text(encoding="utf-8")
    if "[SeqDiag]" in text:
        return 0
    replacements = [
        (
            "        patience = 10\n        while unassigned.numel() >= min_inliers:",
            "        patience = 10\n"
            "        proposal_iteration = 0\n"
            "        while unassigned.numel() >= min_inliers:",
        ),
        (
            "                sample_points=3, inlier_thresh= min(threshold, 0.25)#0.05\n"
            "            )\n\n"
            "            # p, u, theta, phi, mtype",
            "                sample_points=3, inlier_thresh= min(threshold, 0.25)#0.05\n"
            "            )\n"
            "            proposal_iteration += 1\n"
            "            finite_errors = errors[torch.isfinite(errors)]\n"
            "            print(\n"
            "                f\"[SeqDiag] proposal={proposal_iteration} remaining={unassigned.numel()} \"\n"
            "                f\"raw_inliers={inlier_mask.sum().item()} \"\n"
            "                f\"residual_mean={finite_errors.mean().item() if finite_errors.numel() else float('nan'):.8f}\"\n"
            "            )\n\n"
            "            # p, u, theta, phi, mtype",
        ),
        (
            "            if final_mask1.sum().item() == 0:\n"
            "                n_after = unassigned.numel()",
            "            if final_mask1.sum().item() == 0:\n"
            "                print(f\"[SeqDiag] proposal={proposal_iteration} rejected=empty_em_inliers\")\n"
            "                n_after = unassigned.numel()",
        ),
        (
            "            if final_mask.sum().item() < min_inliers:\n"
            "                n_after = unassigned.numel()\n"
            "                continue\n"
            "            #",
            "            if final_mask.sum().item() < min_inliers:\n"
            "                print(\n"
            "                    f\"[SeqDiag] proposal={proposal_iteration} rejected=below_min_inliers \"\n"
            "                    f\"cc_inliers={final_mask.sum().item()} min_inliers={min_inliers}\"\n"
            "                )\n"
            "                n_after = unassigned.numel()\n"
            "                continue\n"
            "            print(f\"[SeqDiag] proposal={proposal_iteration} accepted_inliers={final_mask.sum().item()}\")\n"
            "            #",
        ),
    ]
    for old, new in replacements:
        if old not in text:
            raise RuntimeError(f"Expected AiM source fragment was not found in {path}: {old[:80]!r}")
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
