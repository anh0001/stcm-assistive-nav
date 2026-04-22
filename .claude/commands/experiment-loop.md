---
description: Long-horizon run→diagnose→fix→rerun loop for STCM experiments (single scenario+variant, N cycles)
---

Iterative experiment loop for reviewer evidence. Args: `<scenario> <variant> [max_cycles]`.

Default `max_cycles=5`. Stops early on clean gate or regression.

Mental model: each cycle = one full `run_experiment.py` + diagnosis + one fix.
Evidence accumulates under `results/eval/` — never overwrite prior cycle.

## Cycle protocol

For cycle `i = 1 .. max_cycles`:

### 1. Record baseline (cycle 1 only)
- `git rev-parse HEAD` → baseline SHA
- `git status --short` → flag dirty state in cycle log

### 2. Run
```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 scripts/experiments/run_experiment.py \
  --scenario <scenario> --variant <variant> --skip-bag-hash
```
Capture run dir from stdout. Store as `RUN_I`.

### 3. Parse result.json
Load `results/eval/$RUN_I/result.json`. Extract:
- `failure_flags.*` → any blocker?
- `metrics.f1_1m`, `precision_1m`, `recall_1m`
- `runtime.gdino_ms_p50`, `runtime.sam_ms_p50`

### 4. Diff vs prior cycle
If `i > 1`, compare `RUN_I.result.json` to `RUN_{I-1}.result.json`:
- metric delta (±)
- flag delta (new blocker = regression)
- `git diff RUN_{I-1}_SHA..HEAD -- stcm/ configs/`

### 5. Decide
Use skill recipe table to pick next fix:
- Detection flag → `stcm-tuning` skill
- TF flag → `ros2-debug` skill
- GNG flag → `stcm-tuning` skill
- Silent metric → `silent-failure-hunter` agent
- Clean but metric too low → `stcm-tuning` + consider new variant YAML

If regression vs last cycle: revert last code change (`git checkout HEAD~1 -- <file>`) before applying new fix. Ask user if uncertain.

### 6. Apply fix
- YAML: Edit `stcm/config/semantic_mapping_params.yaml` or create variant in `configs/experiments/variants/<variant>_cycle<i>.yaml`
- Code: Edit `stcm/**`, then `/build`
- Commit each cycle's fix: `git commit -m "exp(cycle<i>): <fix>"` — enables clean diff

### 7. Loop or exit
- If all blocker flags clear AND metric Δ ≥ 0 vs last cycle → mark cycle green. Continue if more metric headroom.
- If 3 consecutive no-improvement cycles → stop, summarize.
- If user `/stop` or hits `max_cycles` → stop, summarize.

## Summary output (end)

Table:
| Cycle | SHA | Flags | f1_1m | Δ vs prev | Fix applied |

+ final `quality_gate.py` output.

## Safety

- Never `rm -rf results/`.
- Never force-push.
- Never mock ROS topics.
- Always run `/build` before rerun if `stcm/` code was edited.
- Always keep cycle `result.json` intact — don't rename/delete.
