# STCM Experiment Configs

This directory contains reviewer-evidence experiment inputs for Codex and shell-driven runs.

- `manifest.yaml` maps scenarios to rosbag2 directories and declares variants/sweeps.
- `variants/*.yaml` are overlays merged onto `stcm/config/semantic_mapping_params.yaml`.
- `ground_truth/*.yaml` is reserved for semantic label and command annotations. Until those files exist, metric scripts still report graph and runtime evidence but mark F1/grounding accuracy as unavailable.

Do not use `output/` files as reviewer evidence directly. Run artifacts belong under `results/eval/` or `results/bench/` with metadata.
