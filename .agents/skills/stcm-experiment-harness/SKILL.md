---
name: stcm-experiment-harness
description: Run STCM reviewer experiments from rosbag2 data with reproducible configs, result JSON, runtime metrics, and fail-loud evidence checks.
origin: project
---

# STCM Experiment Harness

Use this skill when generating evidence for JACIII reviewer items AE-1/R1-1,
AE-3/R1-3, AE-4/R1-4, or AE-9/R2-4.

## Workflow

1. Summarize available bags:

```bash
python3 scripts/experiments/summarize_bags.py
```

2. Dry-run the matrix before running heavy perception:

```bash
python3 scripts/experiments/run_matrix.py --scenario meeting --variant full --no-run --skip-bag-hash
```

3. Run full STCM evidence:

```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 scripts/experiments/run_matrix.py --scenario all --variant full --skip-bag-hash
```

4. Run ablations:

```bash
python3 scripts/experiments/run_matrix.py --scenario all --variant all --skip-bag-hash
```

5. Run sensitivity:

```bash
python3 scripts/experiments/run_matrix.py --scenario meeting --variant full --sweep sensitivity --skip-bag-hash
```

6. Aggregate and gate:

```bash
python3 scripts/experiments/aggregate_results.py
python3 scripts/experiments/quality_gate.py
```

7. Score an existing graph against the meeting-room ground truth:

```bash
python3 scripts/experiments/benchmark_stcm_graph.py \
  --prediction output/stcm.json \
  --ground-truth configs/experiments/ground_truth/meeting_stcm_gt.json \
  --match-threshold-m 1.0
```

## Evidence Rules

- Evidence lives under `results/`, never raw `output/`.
- Every result JSON must include bag path, config snapshot, git SHA, checkpoint
  metadata, launch command, log metrics, graph metrics, benchmark metrics when
  `ground_truth_path` is configured, and failure flags.
- Benchmark scoring uses same-label, one-to-one object-node assignment with XY
  localization error <= `benchmark_match_threshold_m` (default 1.0 m). Treat
  `f1_1m`, `precision_1m`, `recall_1m`, and XY error summaries as the headline
  object-map metrics.
- Treat `zero_detection_frames_gt_10pct`, `zero_object_nodes`,
  `tf_lookup_failures_present`, `gng_update_failures_present`, and
  `benchmark_missing` as blockers until explained.
- Do not merge numbers across different git SHAs unless explicitly reporting
  that as a limitation.
