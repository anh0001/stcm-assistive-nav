# Debugging Rules

See also skill `ros2-debug` (fires on topic/TF/sync symptoms) + `stcm-tuning`
(maps symptom → knob).

## Topic inspection

```bash
ros2 topic echo /semantic_graph/segmented_image
ros2 topic hz <rgb_topic>
ros2 topic info <depth_topic>
```

## Graph visualization (RViz)

1. `rviz2 -d $(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping.rviz`
2. Add MarkerArray display, topic `/semantic_graph/nodes`
3. Set fixed frame to `map` or `world_frame`
4. Nodes = colored spheres w/ labels

## Detection threshold tuning

Too few detections → lower `box_threshold`, `text_threshold` (default 0.55).
Too many false positives → raise them.

## Isolate perception vs ROS

```bash
python3 stcm/test/test_gdino_sam.py <image>
python3 stcm/test/test_depth_anything.py <image>
```

Passing here but not in ROS → sync / TF / topic issue, not model.

Run tests from repo root (scripts use relative paths to `stcm/imgs/` + `stcm/test/`).

## Common issues

- **`use_sim_time` mismatch** — bag without `use_sim_time:=true` → TF lookups fail
- **Frame name drift** — YAML frame ≠ actual TF frame → ImageListener hangs
- **`CameraInfo` missing** — ImageListener blocks forever waiting first msg
- **Empty text prompt class** — each class needs trailing `" ."`
- **Checkpoints missing** — run `/download-ckpts`
- **Stale `install/`** — rerun `/clean-build` after entry-point or setup.py change
- **`PYTHONUSERBASE` unset** — nodes fail to import torch/gdino. Check env before ROS source.

## Testing

```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 stcm/test/test_gdino_sam.py stcm/imgs/irvl-clutter-test.png
python3 stcm/test/test_depth_anything.py stcm/imgs/color-000089.png
python3 stcm/test/ros_test_images.py
```

Tests import `stcm.core` directly, use `stcm/test/_ckpt.py` for workspace
checkpoint dir. Output = annotated images showing detections + seg.
