---
name: stcm-tuning
description: Map STCM symptom to tuning knob. Fire when user says "too few detections", "too many duplicates", "same object split", "nodes drift", "graph exploding", "GNG not converging", "missing objects", "detection thresholds", or asks how to tune box_threshold / text_threshold / target_label_thresholds / GNG params in the semantic_mapping YAML.
---

# STCM Tuning: Symptom → Knob Map

Edit `stcm/config/semantic_mapping_params.yaml` or pass via `--ros-args -p`.

## Detection phase (GroundingDINO)

| Symptom | Knob | Direction |
|---|---|---|
| Missing obvious objects | `box_threshold`, `text_threshold` | Lower (0.55 → 0.3) |
| Too many false positives | same | Raise (0.55 → 0.7) |
| Class never fires | Class not in `text_prompt` OR missing ` .` suffix | Add proper class entry |
| Wrong class picked | `text_prompt` terms ambiguous | Use more specific noun ("office chair" not "chair") |

## Merging phase (spatial dedup)

Per-class radius in `target_label_thresholds` (meters), parallel to
`target_labels`.

| Symptom | Knob | Direction |
|---|---|---|
| Same object = 2+ nodes | `target_label_thresholds[i]` too small | Raise radius |
| Different objects merged | threshold too big | Lower radius |
| Big objects (table) over-split | per-class threshold | Raise (~2.0m) |
| Small objects (cup) merged | per-class threshold | Lower (~0.3m) |

Per `CLAUDE.md`: tables use ~2.0m, chairs ~0.6m — scale w/ object size.

## Instance GNG (i-GNG)

| Symptom | Knob | Direction |
|---|---|---|
| Instances added too eagerly | `gng_min_observations_to_commit` | Raise (e.g. 3 → 5) |
| Outliers creating ghost nodes | `gng_outlier_gate_meters` | Lower |
| Clusters not merging | `gng_cluster_merge_distance` | Raise |
| GNG never stabilizes | `gng_max_age`, `gng_lambda` | Adjust per-label GNG tuning |

Recommend `gng_per_label: true` for mixed object scenes.

## Place GNG (topological)

| Symptom | Knob | Direction |
|---|---|---|
| Too many place nodes | `place_gng_distance_threshold` | Raise |
| Nodes don't connect as path | `place_gng_use_transition_edges` | Enable |
| Semantic scores don't update | `place_gng_update_when_empty` | Enable if detector sparse |
| Stale edges persist | `place_gng_max_edge_age` | Lower |

## Gotchas

1. Tune one knob at a time + re-run offline bag for reproducibility.
2. Use `offline_sequential:=true` for deterministic runs (see
   `rosbag-replay` command).
3. Save graph output under `output/semantic_graph <suffix>.json` so
   comparisons side-by-side possible.
4. Match `target_label_thresholds` length to `target_labels` — mismatch =
   silent fallback, confusing behavior.

## Verification loop

1. Edit YAML
2. `/launch offline_sequential:=true rosbag_path:=...`
3. Inspect `graph_output_path` JSON node count + label distribution
4. Diff vs prior run
