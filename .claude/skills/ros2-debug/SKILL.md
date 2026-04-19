---
name: ros2-debug
description: Diagnose ROS 2 topic/TF/sync problems in STCM pipeline. Fire when user reports "topic not publishing", "no detections appear", "TF missing", "frame doesn't exist", "image_listener stuck", "builder sees nothing", "detections empty", or similar runtime silence from the semantic_map_builder node.
---

# ROS 2 Debug Playbook for STCM

Systematic diagnosis when STCM pipeline runs but produces no output.

## Step 1: Confirm topics publishing

```bash
ros2 topic list | grep -iE "image|depth|camera_info|tf"
ros2 topic hz <rgb_topic>
ros2 topic hz <depth_topic>
ros2 topic echo <camera_info_topic> --once
```

If RGB or depth has 0 Hz → upstream driver/bag issue, not STCM.

## Step 2: Confirm TF chain

Builder needs `camera_frame → base_frame → world_frame`. Check:

```bash
ros2 run tf2_ros tf2_echo <world_frame> <camera_frame>
ros2 run tf2_tools view_frames
```

Missing link = pipeline will silently wait. Common miss: no map→odom
publisher when `world_frame:=map` but only odom exists.

## Step 3: Confirm sync

`message_filters.ApproximateTimeSynchronizer` drops frames if timestamps
diverge. If RGB + depth from different sources, check `use_sim_time` flag
matches bag/live clock source.

## Gotchas (historical failure points)

1. **`use_sim_time` mismatch** — bag replay w/o `use_sim_time:=true` = TF
   lookup fails on past timestamps. Always `true` for bag, `false` for live.
2. **Frame name drift** — YAML `camera_frame` must match actual TF frame
   published by driver. Typo = silent hang in ImageListener.
3. **CameraInfo never received** — ImageListener blocks until first
   CameraInfo. If topic name wrong, node appears alive but processes
   nothing. Check w/ `ros2 topic echo <info_topic> --once`.
4. **Sync tolerance too tight** — RGB/depth at different rates need slop;
   check `ApproximateTimeSynchronizer(slop=...)` value.
5. **GDINO box_threshold too high** — detection returns empty. Lower
   thresholds in YAML (`box_threshold: 0.3`) + retest.
6. **Empty text prompt classes** — each class must end `" ."` (space+dot),
   else GDINO parses wrong.

## Step 4: If all above fine, check segmented_image

```bash
ros2 topic echo /semantic_graph/segmented_image --once
```

Silent = detection empty → tune thresholds (see `stcm-tuning` skill).
Publishing but no graph markers = pose transform failure → check Step 2 TF.

## Escalation

Still stuck? Run `stcm/test/test_gdino_sam.py` against a known-good image to
isolate: ROS issue vs perception-model issue.
