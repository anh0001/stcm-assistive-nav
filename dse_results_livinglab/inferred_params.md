# Livinglab × full-nyu-proposals DSE — Inferred Parameter Space

**Baseline**: F1 = 0.301 ± 0.223 (n=3 historical runs, SHA 67bb403)
**Recent threshold-sweep data**: bt=0.30 → 0.485 (n=3); bt=0.35 → 0.463 (n=3); bt=0.40 → 0.524 (n=1)
**Manifest defaults (livinglab section + full-nyu-proposals.yaml)**:
  - box_threshold: 0.30
  - text_threshold: 0.30
  - gng_min_observations_to_commit: 2
  - cross_label_merge_distance_m: 0.45
  - label_margin_min: 0.05
  - max_observation_range_m: 0.0 (disabled)

| Parameter | Default | Range to explore | Reasoning |
|-----------|---------|------------------|-----------|
| box_threshold | 0.30 | [0.30, 0.35, 0.40, 0.45, 0.50] | existing data shows monotonic 0.30→0.40 improvement; need find peak |
| max_observation_range_m | 0.0 (off) | [0.0, 5.0, 7.0, 10.0] | indoor scene, disabled by default; mid-range gate may cut horizon FPs |
| gng_min_observations_to_commit | 2 | [1, 2, 3, 4] | controls TP/FP trade-off |
| cross_label_merge_distance_m | 0.45 | [0.30, 0.45, 0.60, 0.80] | duplicate suppression radius |
| label_margin_min | 0.05 | [0.03, 0.05, 0.08, 0.12] | CLIP rerank confidence margin |
| text_threshold | 0.30 | [0.25, 0.30, 0.35] | text-image alignment cutoff |

## Strategy
- **Phase 1** (5 iters): broad single-run sweep of top-3 knobs identified from existing data → box_threshold + max_observation_range + gng_min
- **Phase 2** (8 iters): directed combination around best phase-1 point
- **Phase 3** (4 iters): replicate top candidate(s) for variance estimate
