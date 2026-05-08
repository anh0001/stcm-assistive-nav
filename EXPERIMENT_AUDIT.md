# Experiment Audit Report

**Date**: 2026-05-07
**Auditor**: GPT-5.5 xhigh (Codex MCP, read-only)
**Project**: STCM Assistive Navigation — outdoor_livinglab DSE
**Audit target**: F1=0.300 from DSE iter 18 on `outdoor_livinglab × full-nyu-proposals`

## Overall Verdict: **FAIL**

## Integrity Status: **fail**

## Bottom Line

F1=0.300 is an **honest literal number** in the result artifact under the declared scorer. It is **not supported** as:
- "best achievable F1" (only 20 single-run points sampled)
- "real-world outdoor performance" (GT provenance missing)

Measured scoring inflation: **strict_same_label F1 = 0.250 vs declared alias F1 = 0.300 → 1.20× inflation**. Inflation vs true hand-annotated GT: unknown.

## Checks

### A. Ground Truth Provenance — **FAIL**

`configs/experiments/ground_truth/outdoor_livinglab_stcm_gt.json` starts as an STCM container (line 2). Contains `runtime` events/timings (line 2805) — top keys are only `llm`, `metadata`, `place_graph`, `semantic_graph`, `stcm_version`. **No `_provenance` block, no annotator, no frozen-date, no inclusion protocol**.

Manifest description says only "Ground truth frozen v1.0 from outdoor_livinglab_stcm_gt.json" (`manifest.yaml:232`).

Comparison: livinglab GT has annotation protocol metadata; outdoor lacks it.

**Evidence this is a frozen STCM output, not hand-annotated**: presence of `runtime.events.{frames_seen, gng_update_calls, pose_failures, raw_detections, target_detections, ...}` at GT line 2805. Hand annotation would not produce GNG/perception-runtime telemetry.

**Verdict**: treat as frozen STCM-shaped target, not real GT, until provenance is added.

### B. Score Normalization — **WARN**

`benchmark_stcm_graph.py`: TP/FP/FN computation is conventional Hungarian matching with thresholding (lines 176, 180, 208, 211).
- `tp = len(matches)`, `fp = len(false_positive_nodes)`, `fn = len(false_negative_gt_nodes)`
- Precision denom = `tp+fp` (valid pred nodes after invalid-pose filter at line 477)
- Recall denom = `tp+fn`

No fraud. But: `_node_pose` (`benchmark_stcm_graph.py:67`) does not finite-check NaN/inf poses. For the iter 18 run, `invalid_prediction_nodes=0`, so no current numeric effect.

### C. Result File Existence and Number Match — **PASS**

- `results/eval/20260507_073408_outdoor_livinglab_full-nyu-proposals.json:1279` contains `tp:6 fp:15 fn:13 f1:0.3`. ✓
- `dse_results/dse_log.csv:19` row 18 references same file with same numbers. ✓
- `dse_results/outputs/iter_018/stdout.log:1` confirms write. ✓
- Graph audit: 21 nodes, max abs pose 12.67m (no 1e10+ artifacts). ✓

**Fragility flag**: `run_dse_point.py:157` selects newest matching result file after subprocess completes — a failed/no-output run could log a stale file's metrics. No bug observed in iter 18, but harness is brittle.

### D. Dead Knobs / Dead Code — **FAIL**

DSE_REPORT claims `box_threshold`, `text_threshold`, `target_label_thresholds[trailer]`, `label_margin_min` are all dead. Audit found:

| Knob | DSE claim | Actual code reality |
|---|---|---|
| `box_threshold` / `text_threshold` | dead (NYU bank overrides) | **Confirmed dead**: `nyu_grounded_backend.py:237` overrides chunk/global thresholds w/ per-class values; passed to GDINO at line 421. |
| `target_label_thresholds[trailer]` | dead (NYU bank overrides) | **Confirmed dead, but reason mis-stated**: dead because GNG manager receives `gng_cluster_merge_distance`/`cross_label_merge_distance_m`, NOT per-label thresholds (`semantic_map_builder.py:315`). Not a NYU-bank override. |
| `label_margin_min` | dead | **NOT DEAD**: `nyu_grounded_backend.py:380` uses it in `choose_label`. The two tested values (0.05, 0.10) just happened to tie on this scene. False conclusion in DSE_REPORT. |

DSE swept real knobs and no-ops together. Phase 2 was largely wasted.

### E. Scope Assessment — **FAIL**

DSE budget = 20 cycles (`inferred_params.md:5`). DSE_REPORT calls iter 18 "Best found" (`DSE_REPORT.md:13`) but recommendations text implies "adopt as default" (line 80 admits multi-seed averaging is still needed).

Eval protocol contract (`eval-protocol.md:7, :122`) requires:
- ≥3 independent runs per scenario w/ fresh seeds
- Reproducibility: git SHA, dirty flag, bag sha256, seed, hostname

iter 18 violates:
- single seed
- dirty git state at run time (default behavior of `run_experiment.py`)
- skipped bag file hash (`--skip-bag-hash`)
- no recorded seed/hostname

**Phrase "BEST ACHIEVABLE F1" is scope inflation.** Supported phrase: "best found among 20 single-run DSE points on one scene".

### F. Label Aliasing Sanity — **FAIL**

Aliases are GT-label-keyed and directional (`benchmark_stcm_graph.py:243, :266`). Manifest aliases let:
- `trolley` GT accept `electric utility vehicle` pred (`manifest.yaml:292`)
- `rectangular table` GT accept `trailer` pred (`manifest.yaml:308`)

No many-to-one TP double counting (Hungarian + `matched_pred_ids` at `benchmark_stcm_graph.py:289`). But the alias set itself **inflates the score**:

| Scoring mode | TP | FP | FN | F1 |
|---|---|---|---|---|
| Strict same-label, 1.0m match | 5 | 16 | 14 | **0.250** |
| Declared alias + 1.5m match | 6 | 15 | 13 | **0.300** |

**Inflation = 1.20×.** The extra TP is `trolley_1_0` matched to `electric utility vehicle_inst_5` at 1.117m via alias.

`rectangular_table → trailer` alias is the most suspect — those are visually + semantically distinct objects.

### G. Range Gate Implementation Bug — **FAIL**

`semantic_map_builder.py:1607`: range gate computes `obs_range = np.linalg.norm(pose_xy - robot_xy)` then checks only `obs_range > max`. **No `np.isfinite` guard.**

NaN behavior: `NaN > 5.0` evaluates **False** → NaN pose survives the gate. GNG accepts `centroid_arr = np.asarray(...).reshape(3)` without finite validation (`gng_instance_manager.py:168`).

iter 18's specific graph has no huge/non-finite poses, so number is not affected — but the **code path is unsafe and must be fixed before any paper claim**.

## F. Evaluation Type Classification

**`synthetic_proxy`** (downgrade from claimed `real_gt`).

Outdoor GT is a frozen STCM-shaped target with no hand-annotation provenance, perception-runtime telemetry intact in metadata, and no annotator/protocol record. Cannot be classified as `real_gt` until provenance is added.

## Action Items (in priority order)

1. **Fix outdoor GT provenance** — add `_provenance` block to `outdoor_livinglab_stcm_gt.json`: annotator, frozen date, source bag, annotation protocol, inclusion criteria, pose derivation, and explicit statement of whether any STCM output was used as a draft. **Blocks any paper claim using S3.**

2. **Fix range gate NaN hole** in `stcm/stcm/nodes/semantic_map_builder.py:1607` — reject `not np.isfinite(pose).all()` before range check + add NaN-pose regression test in `stcm/test/`.

3. **Report both scoring modes** in DSE_REPORT and any paper table:
   - `strict_same_label_F1@1.0m` (the indoor protocol)
   - `alias_F1@1.5m` (the relaxed outdoor protocol)
   - **Remove or justify** `rectangular table → trailer` and the vehicle cross-aliases.

4. **Reword DSE claims**: replace "best achievable" / "BEST" language with "best found among 20 single-run DSE points on 1 scene". Re-run iter 18 with **≥3 seeds on clean git and recorded bag SHA-256** before committing as paper config.

5. **Harden harness**:
   - `run_dse_point.py:157`: capture exact result path from `run_experiment.py` stdout or pass an explicit run id; fail if no new result file produced.
   - Reject runs with dirty git unless `--allow-dirty` flag.
   - Force record seed + hostname in result.json (already in eval-protocol contract).

## Claim Impact

| Claim | Original | After audit |
|---|---|---|
| C1: F1=0.300 on outdoor S3 | supported | **needs qualifier**: "with declared label aliases, 1.5m match threshold, 1 scene, 1 seed, against GT of unverified provenance" |
| C2: best achievable from 20-cycle DSE | supported | **needs qualifier**: "best found, not proven optimum; many phase-2 knobs were no-ops" |
| C3: outdoor capability demonstrated | supported | **unsupported** until GT provenance fixed |
| C4: range gate filters horizon FP correctly | supported | **needs qualifier**: "for finite poses only; NaN poses bypass gate" |

## Inflation Summary

| Source | Lift |
|---|---|
| Label aliasing (alias vs strict same-label) | +20% (0.250 → 0.300) |
| Match threshold relaxation (1.5m vs 1.0m) | bundled into above |
| GT provenance | unknown (real GT could be much harder OR easier) |
| Single-seed run-to-run variance | ±5-10% F1 (per session evidence) |
