# Experiment Harness Rules

Authoritative rules for STCM experiment runner (parity w/ Codex
`.agents/skills/stcm-experiment-harness`). Skill `stcm-experiment-harness`
+ commands `/experiment`, `/experiment-matrix`, `/experiment-loop` point here.

## Scripts (invoked from repo root)

| Script | Purpose |
|--------|---------|
| `scripts/experiments/summarize_bags.py` | List rosbag metadata: path, topics, duration, hash |
| `scripts/experiments/run_experiment.py` | Single deterministic offline bag replay → result JSON |
| `scripts/experiments/run_matrix.py` | Fan-out runner: scenarios × variants × sensitivity |
| `scripts/experiments/aggregate_results.py` | Rebuild CSV summaries from `results/eval/*/result.json` |
| `scripts/experiments/benchmark_stcm_graph.py` | Score one graph JSON vs GT JSON (same-label 1:1, 1m) |
| `scripts/experiments/quality_gate.py` | Evidence readiness gate — exit non-zero on blockers |

All scripts require env sourced:
```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
source /opt/ros/humble/setup.bash
source install/setup.bash
```

## Configs

- `configs/experiments/manifest.yaml` — scenario bags, variants, required
  topics, sensitivity values
- `configs/experiments/variants/*.yaml` — frozen ablation YAML
- `configs/experiments/ground_truth/<scene>_stcm_gt.json` — GT graph per scene

## Result layout (output contract)

```
results/
├── eval/
│   └── <scenario>_<variant>_<sha>_<timestamp>/
│       ├── config.yaml       # merged YAML snapshot
│       ├── command.txt       # launch command
│       ├── launch.log        # stdout/stderr tail
│       ├── graph.json        # STCM output graph
│       ├── runtime.json      # per-module timings
│       └── result.json       # evidence record
└── bench/
    └── runtime_<sha>.json
```

### result.json schema (required fields)

- `scenario`, `variant`, `sensitivity` (or null)
- `bag_path`, `bag_sha256` (or `"skipped"` + flag recorded)
- `git_sha`, `git_dirty` (bool)
- `checkpoints`: {`gdino_sha256`, `mobilesam_sha256`, `depth_anything_sha256?`}
- `command`, `log_tail`
- `metrics`: `{f1_1m, precision_1m, recall_1m, xy_err_mean, xy_err_p95, label_cov, ...}`
- `runtime`: `{gdino_ms_p50, gdino_ms_p95, sam_ms_p50, ...}`
- `failure_flags`:
  - `zero_object_nodes`
  - `zero_detection_frames_gt_10pct`
  - `tf_lookup_failures_present`
  - `gng_update_failures_present`
  - `benchmark_missing`
- `seed`, `hostname`, `wall_clock_utc`

## Blocker policy

Any flag true = blocker. Do not aggregate blocked runs into paper tables
until resolved or explicitly noted as ablation observation.

Gate order:
1. `benchmark_missing` first — fix GT path, rerun scorer alone.
2. `zero_object_nodes` / `tf_lookup_failures_present` — ROS-level issue;
   load `ros2-debug` skill.
3. `zero_detection_frames_gt_10pct` / `gng_update_failures_present` —
   knob-level; load `stcm-tuning` skill.

## Long-horizon loop contract

`/experiment-loop` binds these rules:
- One fix per cycle, committed separately (`exp(cycleN): <desc>`).
- Never overwrite prior `result.json`. Evidence accumulates.
- Cycle stops if 3 consecutive no-improvement OR user stops OR `max_cycles`.
- Regression detection: if metric strictly worse than prior cycle AND no new
  flags resolved → consider revert before applying next fix.

## Parity with Codex harness

Claude side and Codex side must stay compatible:
- Same scripts under `scripts/experiments/`
- Same manifest YAML
- Same `result.json` schema
- Same GT layout

Divergence = silent drift. If schema changes: update both
`.claude/rules/experiment-harness.md` and
`.agents/skills/stcm-experiment-harness/SKILL.md` in same PR.

## Cross-refs

- Eval scenarios + metrics → `.claude/rules/eval-protocol.md`
- Runtime metrics → `.claude/rules/benchmark-protocol.md`
- Env setup → `.claude/rules/env-setup.md`
- Build/launch → `.claude/rules/build-and-launch.md`
- Debug paths → `.claude/rules/debugging.md`
