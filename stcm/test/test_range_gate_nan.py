#!/usr/bin/python3
"""Regression test: range gate must reject non-finite poses.

Audit (2026-05-07) flagged that the original gate at semantic_map_builder.py
checked only `obs_range > max_observation_range_m`. NaN comparisons return
False in numpy, so a NaN pose would silently survive the gate and corrupt
the GNG centroid average.

This test verifies the corrected gating predicate handles NaN, +inf, -inf,
and bad robot xy.

Run:
    python3 stcm/test/test_range_gate_nan.py
"""
from __future__ import annotations

import sys

import numpy as np


def gate_decision(pose, robot_xyz, max_range_m):
    """Replicate the gating predicate used in semantic_map_builder.

    Returns "reject_nonfinite_pose" / "skip_gate_nonfinite_robot" /
    "reject_out_of_range" / "accept".
    """
    pose_arr = np.asarray(pose, dtype=float)
    if pose_arr.size < 2 or not np.isfinite(pose_arr[:2]).all():
        return "reject_nonfinite_pose"
    if max_range_m <= 0.0 or robot_xyz is None:
        return "accept"
    robot_xy = np.asarray(robot_xyz, dtype=float)[:2]
    if not np.isfinite(robot_xy).all():
        return "skip_gate_nonfinite_robot"
    obs_range = float(np.linalg.norm(pose_arr[:2] - robot_xy))
    if not np.isfinite(obs_range) or obs_range > max_range_m:
        return "reject_out_of_range"
    return "accept"


def main() -> int:
    cases = [
        # (label, pose, robot_xyz, max_range, expected)
        ("finite within range", [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], 5.0, "accept"),
        ("finite at exact range", [3.0, 4.0, 0.0], [0.0, 0.0, 0.0], 5.0, "accept"),
        ("finite just outside",   [3.0, 4.1, 0.0], [0.0, 0.0, 0.0], 5.0, "reject_out_of_range"),
        ("NaN x",                 [np.nan, 1.0, 0.0], [0.0, 0.0, 0.0], 5.0, "reject_nonfinite_pose"),
        ("NaN y",                 [1.0, np.nan, 0.0], [0.0, 0.0, 0.0], 5.0, "reject_nonfinite_pose"),
        ("+inf x",                [np.inf, 1.0, 0.0], [0.0, 0.0, 0.0], 5.0, "reject_nonfinite_pose"),
        ("-inf y",                [1.0, -np.inf, 0.0], [0.0, 0.0, 0.0], 5.0, "reject_nonfinite_pose"),
        ("astronomically large",  [1e24, 1e24, 0.0], [0.0, 0.0, 0.0], 5.0, "reject_out_of_range"),
        ("NaN robot",             [1.0, 1.0, 0.0], [np.nan, 0.0, 0.0], 5.0, "skip_gate_nonfinite_robot"),
        ("max_range disabled",    [1e24, 1e24, 0.0], [0.0, 0.0, 0.0], 0.0, "accept"),
        ("robot None disables",   [1e24, 1e24, 0.0], None, 5.0, "accept"),
    ]
    failed = 0
    for label, pose, robot, mr, expected in cases:
        got = gate_decision(pose, robot, mr)
        ok = got == expected
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: expected={expected} got={got}")
        if not ok:
            failed += 1
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
