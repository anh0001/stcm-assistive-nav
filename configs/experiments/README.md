# STCM Experiment Configs

This directory contains reviewer-evidence experiment inputs for Codex and shell-driven runs.

- `manifest.yaml` maps scenarios to rosbag2 directories and declares variants/sweeps.
- `variants/*.yaml` are overlays merged onto `stcm/config/semantic_mapping_params.yaml`.
- `ground_truth/` stores scenario ground-truth inputs, including STCM JSON graph
  fixtures such as `meeting_stcm_gt.json` and optional YAML semantic label or
  command annotations. Scenario entries should reference these files with
  `ground_truth_path`.
- `scripts/experiments/benchmark_stcm_graph.py` scores a predicted STCM JSON
  against a ground-truth STCM JSON using same-label, one-to-one object matching
  at the scenario's `benchmark_match_threshold_m` XY distance.

Do not use `output/` files as reviewer evidence directly. Run artifacts belong under `results/eval/` or `results/bench/` with metadata.
