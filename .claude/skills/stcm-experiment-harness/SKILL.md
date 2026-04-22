---
name: stcm-experiment-harness
description: Run STCM reviewer experiments from rosbag2 data with reproducible configs, result JSON, runtime metrics, and fail-loud evidence checks. Supports long-horizon run→diagnose→fix→rerun loops over AE-1/R1-1, AE-3/R1-3, AE-4/R1-4, AE-9/R2-4.
origin: project
tools: Read, Write, Edit, Bash, Grep, Glob
---

# STCM Experiment Harness (Claude parity with Codex)

Mirror of `.agents/skills/stcm-experiment-harness` so Claude Code can run
experiments, inspect failure flags, apply code/config fixes, and rerun —
all without leaving the conversation.

## When to activate

- Reviewer evidence runs (single scenario, matrix, ablation, sensitivity)
- Long-horizon loops: run → parse `results/` → fix bug/tune knob → rerun
- Offline rosbag replay with deterministic seed + config snapshot
- Any task where evidence must land under `results/`, not `output/`

## Environment (always before running)

```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
source /opt/ros/humble/setup.bash
source install/setup.bash
```

## Canonical workflow

### 1. Preflight (no perception load)

```bash
python3 scripts/experiments/summarize_bags.py
python3 scripts/experiments/run_matrix.py \
  --scenario meeting --variant full --no-run --skip-bag-hash
```

Check: bag path resolves, required topics present, merged YAML prints clean.

### 2. Single experiment

```bash
python3 scripts/experiments/run_experiment.py \
  --scenario meeting --variant full --skip-bag-hash
```

Writes `results/eval/<scenario>_<variant>_<sha>_<ts>/` with:
- `config.yaml` (merged snapshot)
- `command.txt` + `launch.log`
- `graph.json` (STCM output), `runtime.json` (per-module timings)
- `result.json` (bag hash, ckpt hashes, git SHA, failure flags, metrics)

### 3. Matrix (scenarios × variants × sensitivity)

```bash
# All scenarios, full variant
python3 scripts/experiments/run_matrix.py --scenario all --variant full --skip-bag-hash

# Full ablation matrix
python3 scripts/experiments/run_matrix.py --scenario all --variant all --skip-bag-hash

# Sensitivity sweep (Table E)
python3 scripts/experiments/run_matrix.py \
  --scenario meeting --variant full --sweep sensitivity --skip-bag-hash
```

### 4. Aggregate + gate

```bash
python3 scripts/experiments/aggregate_results.py
python3 scripts/experiments/quality_gate.py
```

`quality_gate.py` exit non-zero → blockers present.

### 5. Score isolated graph against GT

```bash
python3 scripts/experiments/benchmark_stcm_graph.py \
  --prediction output/stcm.json \
  --ground-truth configs/experiments/ground_truth/meeting_stcm_gt.json \
  --match-threshold-m 1.0
```

## Long-horizon run→fix→rerun loop

Core use case: experiment fails or gate blocks → diagnose → fix code/YAML →
rerun same scenario+variant → compare. Steps per cycle:

1. **Run** current cycle via `run_experiment.py` or `run_matrix.py`. Capture
   run dir path from stdout.
2. **Parse** `results/eval/<run_dir>/result.json`:
   - `failure_flags.zero_object_nodes` → detection/TF broken
   - `failure_flags.zero_detection_frames_gt_10pct` → prompt or threshold
   - `failure_flags.tf_lookup_failures_present` → frame/sim-time mismatch
   - `failure_flags.gng_update_failures_present` → GNG params
   - `failure_flags.benchmark_missing` → GT path or scorer bug
   - `metrics.f1_1m`, `metrics.precision_1m`, `metrics.recall_1m`
3. **Decide knob or code change**:
   - Detection-side → `stcm-tuning` skill (`box_threshold`, `text_threshold`,
     `target_label_thresholds`)
   - GNG-side → `stcm-tuning` skill (`gng_*`, `place_gng_*`)
   - TF / sync → `ros2-debug` skill
   - Silent-failure / metric fake → `silent-failure-hunter` agent
4. **Apply fix**:
   - Code: `Edit` stcm/ source, then `/build`
   - YAML: `Edit` `stcm/config/semantic_mapping_params.yaml` or create
     variant under `configs/experiments/variants/`
5. **Rerun**: same CLI, new run dir written. Never overwrite prior cycle.
6. **Diff**: compare `result.json` metrics + failure_flags between cycles.
   If worse, consider `git diff` to revert.
7. **Checkpoint**: once `quality_gate.py` passes clean on target scenario,
   lock in + move to next reviewer item.

## Invariants (fail loud, never paper over)

- Evidence lives under `results/`. Raw `output/` is scratch only.
- Every `result.json` must carry: bag path, bag sha256 (or `--skip-bag-hash`
  flag recorded), merged config YAML, git SHA, git dirty flag, checkpoint
  sha256s, command string, log tail, metrics, failure flags.
- Benchmark scoring = same-label 1:1 object-node assignment, XY error ≤
  `benchmark_match_threshold_m` (default 1.0m). Headline metrics: `f1_1m`,
  `precision_1m`, `recall_1m`, XY summary.
- Blockers until explained: `zero_detection_frames_gt_10pct`,
  `zero_object_nodes`, `tf_lookup_failures_present`,
  `gng_update_failures_present`, `benchmark_missing`.
- Never merge numbers across git SHAs without noting limitation.
- Never mock ROS topics for reviewer numbers.
- If FAST-LIO2 TF gap > 0.2s → discard run.
- If GDINO returns 0 detections for >10% frames → flag; keep for ablation note.

## Recovery recipes (common failure → fix)

| Flag | First suspect | Fix command |
|------|---------------|-------------|
| `zero_object_nodes` | TF chain broken | load `ros2-debug` skill; check `camera→base→world` |
| `zero_detection_frames_gt_10pct` | prompt or thresholds | lower `box_threshold`/`text_threshold`, check class `" ."` suffix |
| `tf_lookup_failures_present` | sim-time mismatch | set `use_sim_time:=true` for bag; verify bag publishes `/clock` |
| `gng_update_failures_present` | GNG init / params | load `stcm-tuning`; inspect `gng_min_observations_to_commit` |
| `benchmark_missing` | no GT path or scorer error | verify `configs/experiments/ground_truth/<scene>_stcm_gt.json` exists; rerun `benchmark_stcm_graph.py` standalone |
| metric regressed vs prior cycle | recent edit | `git diff HEAD~1 -- stcm/`; consider revert |

## Pointers

- Rules: `.claude/rules/experiment-harness.md` (authoritative)
- Eval scenarios / metrics: `.claude/rules/eval-protocol.md`
- Runtime metrics: `.claude/rules/benchmark-protocol.md`
- Commands: `/experiment`, `/experiment-matrix`, `/experiment-loop`
- Codex counterpart: `.agents/skills/stcm-experiment-harness/SKILL.md`
