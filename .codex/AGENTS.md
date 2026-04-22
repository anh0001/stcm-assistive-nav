# Codex Experiment Supplement

This file adapts useful `.claude` assets for Codex work in this repo. The root
`AGENTS.md` remains the primary instruction file.

Codex layout:
- `.agents/skills/` contains reusable project-local skills.
- `.codex/config.toml` contains Codex CLI config and role definitions.
- `.codex/agents/*.toml` contains local multi-agent role configs.

## Adopted `.claude` Assets

- `.claude/rules/eval-protocol.md` -> `configs/experiments/manifest.yaml`,
  `scripts/experiments/run_experiment.py`, and `run_matrix.py`.
- `.claude/rules/benchmark-protocol.md` -> runtime metadata saved by
  `semantic_map_builder` and aggregated by `aggregate_results.py`.
- `.claude/commands/rosbag-replay.md` -> `run_experiment.py` preflight and
  offline sequential launch command generation.
- `.claude/commands/quality-gate.md` -> `scripts/experiments/quality_gate.py`
  for experiment evidence readiness.
- `.claude/skills/stcm-tuning/SKILL.md` -> frozen variant YAMLs and sensitivity
  sweep keys, plus `.agents/skills/stcm-tuning`.
- `.claude/skills/ros2-debug/SKILL.md` -> required topic checks and fail-loud
  result flags, plus `.agents/skills/stcm-ros2-debug`.

## Codex Defaults

- Treat experiments as the main workstream. Paper editing is out of scope unless
  explicitly requested.
- Never use `output/` as evidence directly. Evidence lives under `results/`.
- For scenarios with `ground_truth_path`, `run_experiment.py` must produce
  same-label object-map benchmark metrics (`precision_1m`, `recall_1m`,
  `f1_1m`, and XY error summaries) under the run artifact directory.
- Prefer the `.agents/skills/stcm-experiment-harness` workflow when running
  reviewer experiments.
- Prefer dry-run first:

```bash
python3 scripts/experiments/run_matrix.py --scenario meeting --variant full --no-run --skip-bag-hash
```

- To score a manually produced graph:

```bash
python3 scripts/experiments/benchmark_stcm_graph.py \
  --prediction output/stcm.json \
  --ground-truth configs/experiments/ground_truth/meeting_stcm_gt.json \
  --match-threshold-m 1.0
```

- After real runs, rebuild summaries and inspect the gate:

```bash
python3 scripts/experiments/aggregate_results.py
python3 scripts/experiments/quality_gate.py
```
