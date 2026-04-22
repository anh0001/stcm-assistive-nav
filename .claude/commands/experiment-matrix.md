---
description: Run STCM experiment matrix (scenarios × variants × sensitivity), aggregate, quality gate
---

Matrix-run wrapper. Args: `[scenario] [variant] [--sweep sensitivity]`.

Scenario default `all`. Variant default `full`. Sweep optional.

Steps:
1. Load `stcm-experiment-harness` skill.
2. Dry-run preflight:
   ```bash
   python3 scripts/experiments/run_matrix.py \
     --scenario $SCENARIO --variant $VARIANT $SWEEP --no-run --skip-bag-hash
   ```
3. Source env (see skill) + execute matrix:
   ```bash
   python3 scripts/experiments/run_matrix.py \
     --scenario $SCENARIO --variant $VARIANT $SWEEP --skip-bag-hash
   ```
4. Aggregate + gate:
   ```bash
   python3 scripts/experiments/aggregate_results.py
   python3 scripts/experiments/quality_gate.py
   ```
5. Report: per-run `result.json` summary table (scenario, variant, f1_1m, flags).
6. If `quality_gate.py` non-zero → list blockers + map each to skill recovery recipe.
7. If all green → suggest `/verify` for paper preflight.

Paper target: Table B (nav+F1), Table C (LLM ablation), Table E (sensitivity).
