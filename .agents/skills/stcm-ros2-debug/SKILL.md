---
name: stcm-ros2-debug
description: Diagnose STCM ROS 2 topic, TF, clock, sync, and silent-output failures during offline rosbag or live runs.
origin: project
---

# STCM ROS 2 Debug

Use this when a run produces no graph, no detections, missing poses, TF errors,
or missing segmented images.

## Checks

```bash
ros2 bag info <bag_path>
ros2 topic list | grep -iE "image|camera_info|tf|lidar"
ros2 topic hz <rgb_topic>
ros2 topic echo <camera_info_topic> --once
ros2 run tf2_ros tf2_echo <world_frame> <camera_frame>
```

## Common Causes

- `use_sim_time` false for bag replay.
- YAML frame names do not match TF.
- CameraInfo topic is wrong or absent.
- Projected LiDAR topic lacks `u`/`v` fields.
- GroundingDINO prompt classes are missing the trailing `" ."`.
- Detection thresholds are too high.
- `PYTHONUSERBASE` was not exported before sourcing ROS.

Prefer `scripts/experiments/run_experiment.py --no-run` first to confirm the
exact generated launch command and config snapshot.

