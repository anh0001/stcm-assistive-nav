---
description: Run STCM paper eval protocol (navigation + semantic + grounding) for revision Tables B/C/E
---

Drive eval-harness skill for JACIII Jc26-0002 revision (AE-1/R1-1, AE-3/R1-3, AE-9/R2-4).

Args: `[scenario]` = `corridor` | `office` | `meeting` | `all` (default `all`).

Steps:
1. Load eval-harness skill.
2. Emit/execute YAML tasks per scenario × ≥3 independent runs:
   - **nav_success**: planned goal reached w/ Nav2; metric = success rate + 95% CI, completion time, path-length ratio.
   - **semantic_f1**: macro/micro-F1 on stable place-node labels vs ground-truth vocab (`corridor_labels.yaml`).
   - **grounding_accuracy**: exact-match + top-2 on ≈15 commands/scenario (simple / disambiguation / compositional).
3. Paired ablations on same data: `full` vs `semantic-only` vs `place-gng-only` vs `no-llm`. McNemar test per command subset.
4. Emit results to `results/eval/<scenario>_<run_id>.json`. Aggregate into Table B (nav+F1) + Table C (LLM ablation).
5. Never mock ROS topics — real rosbag replay or live run. Fail loud if FAST-LIO2/Nav2/GDINO silent.

Sensitivity sweep (Table E): pass `--sweep` to parametrize `D_new ∈ {0.3,0.5,0.8}`, `α ∈ {0.05,0.1,0.2}`, `τ_box=τ_text ∈ {0.45,0.55,0.65}`, `η_min ∈ {1,3,5}`.

Reproducibility: fix all seeds; record commit SHA, rosbag path, checkpoint hashes in result JSON.
