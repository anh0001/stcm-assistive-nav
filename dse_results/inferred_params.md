# Inferred Parameter Space — DSE Outdoor S3

**Scenario**: outdoor_livinglab
**Variant**: full-nyu-proposals
**Objective**: maximize F1 (benchmark `match_threshold_m=1.5m`, GT=19 nodes)
**Budget**: 20 cycles (~2.7h wall-clock at ~8min/run)

## Knobs Selected

Picked from prior debugging session showing strong sensitivity:

| Parameter | Source | Default (current) | Inferred Range | Reasoning |
|-----------|--------|------------------|---------------|-----------|
| `max_observation_range_m` | scenario config_overrides | 5.0 | [3, 4, 5, 6, 7, 10] | Range gate cuts horizon FPs; sweep around current best; F1 climbed +57% from 0→5 |
| `gng_min_observations_to_commit` | scenario config_overrides | 2 | [1, 2, 3] | Filters transient detections; bigger = fewer FP but kills recall |
| `box_threshold` / `text_threshold` | scenario config_overrides | 0.30 / 0.30 | [0.25, 0.28, 0.30, 0.32, 0.35] | Detector confidence cutoff; tied together |
| `target_label_thresholds[trailer]` | scenario config_overrides | 1.5 | [1.2, 1.5, 1.8, 2.0] | Hard cap 2.29m (min trailer-pair GT separation); below 1.5 splits same instance |
| `label_margin_min` | scenario config_overrides | 0.05 | [0.03, 0.05, 0.08] | Score margin between top-2 label scores |
| `cross_label_merge_distance_m` | scenario config_overrides | 0.45 | [0.30, 0.45, 0.60] | Inter-label cosine merge distance |

## Knobs FROZEN (not swept)

- `nyu_prompt_bank_path`: outdoor NYU prompt bank — locked
- `instance_label_switch_min_observations`: 2 — locked
- `cross_label_merge_min_cosine`: 0.20 — locked
- `target_labels`: 14 outdoor classes — locked
- `target_label_thresholds[non-trailer]`: locked at current defaults

## Search Strategy

- **Phase 1 (8 runs)**: sweep `max_observation_range_m` × `gng_min_obs` Cartesian → best `(R, G)`.
- **Phase 2 (8 runs)**: hold best `(R, G)`, sweep `box_threshold==text_threshold` × `target_label_thresholds[trailer]`.
- **Phase 3 (4 runs)**: refine around top configuration with `label_margin_min` and `cross_label_merge_distance_m` perturbations.

## Boundary Expansion

If best `max_observation_range_m` lands at boundary (3 or 10), extend by 1 step (2 or 12) on next phase.

## Baseline (iteration 0)

Already established from session run `203016_tuned`:
- F1 = 0.172 (tuned variant)
- F1 = 0.192 (full variant, run `202020`)

We sweep `outdoor_livinglab` × `full-nyu-proposals`, so baseline = **F1=0.192** (run `202020_outdoor_livinglab_full-nyu-proposals`).
