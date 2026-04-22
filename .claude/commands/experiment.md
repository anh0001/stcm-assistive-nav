---
description: Run single STCM experiment (scenario+variant) with reproducible evidence under results/
---

Single-run experiment wrapper. Args: `[scenario] [variant]`.

Scenarios: `corridor` | `office` | `meeting` | `all`. Default `meeting`.
Variants: `full` | `semantic-only` | `place-gng-only` | `no-llm` | `all`. Default `full`.

Steps:
1. Load `stcm-experiment-harness` skill.
2. Preflight dry-run:
   ```bash
   python3 scripts/experiments/run_matrix.py \
     --scenario $SCENARIO --variant $VARIANT --no-run --skip-bag-hash
   ```
   Abort if bag missing or required topics fail.
3. Source env then execute:
   ```bash
   export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
   source /opt/ros/humble/setup.bash
   source install/setup.bash
   python3 scripts/experiments/run_experiment.py \
     --scenario $SCENARIO --variant $VARIANT --skip-bag-hash
   ```
4. Parse freshest `results/eval/<scenario>_<variant>_*/result.json`. Print:
   - metrics (`f1_1m`, `precision_1m`, `recall_1m`, xy err)
   - all `failure_flags` states
5. If any blocker flag true → print recovery recipe from skill recipe table + stop.
6. If clean → suggest next action (`/experiment-matrix` or `/experiment-loop`).

Never mock topics. Never treat `output/` as evidence.
