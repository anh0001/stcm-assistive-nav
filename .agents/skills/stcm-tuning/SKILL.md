---
name: stcm-tuning
description: Tune STCM detection thresholds, object merging, instance GNG, and place GNG from observed experiment symptoms.
origin: project
---

# STCM Tuning

Use this when a graph has too few detections, too many false positives,
duplicate objects, over-merged objects, sparse place nodes, or unstable GNG.

## Detection

| Symptom | Knob | Direction |
|---|---|---|
| Missing objects | `box_threshold`, `text_threshold` | Lower |
| False positives | `box_threshold`, `text_threshold` | Raise |
| Class never fires | `text_prompt` | Ensure each class ends `" ."` |
| Wrong class picked | `text_prompt` | Use more specific object phrases |

## Object Merging

`target_label_thresholds` must match `target_labels` length.

| Symptom | Knob | Direction |
|---|---|---|
| Same object split | per-label threshold | Raise |
| Different objects merged | per-label threshold | Lower |
| Outlier objects | `gng_outlier_gate_meters` | Enable/lower |
| Premature instances | `gng_min_observations_to_commit` | Raise |

## Place GNG

| Symptom | Knob | Direction |
|---|---|---|
| Too many place nodes | `place_gng_distance_threshold` | Raise |
| Too few place nodes | `place_gng_distance_threshold` | Lower |
| Weak semantic labels | `place_gng_semantic_alpha` | Adjust sweep |
| Missing path edges | `place_gng_use_transition_edges` | Enable |

Use `run_matrix.py --sweep sensitivity` for reviewer-safe tuning evidence.

