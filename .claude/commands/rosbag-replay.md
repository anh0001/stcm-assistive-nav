---
description: Launch semantic_mapping against a rosbag2 in offline sequential mode
argument-hint: <rosbag_path> [storage_id]
---

Offline deterministic replay of rosbag2 through STCM pipeline.

Required arg: `$ARGUMENTS` = rosbag dir path. Optional 2nd arg = storage_id (default `sqlite3`).

Invocation:
```bash
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=$(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping_params.yaml \
  offline_sequential:=true \
  rosbag_path:=<path> \
  rosbag_storage_id:=<storage> \
  use_sim_time:=true
```

Pre-check `ros2 bag info <path>` to confirm topics exist + frame count. Warn if
expected topics (`rgb_topic`, `depth_topic`, `camera_info_topic` from YAML) not
present in bag.

Why offline_sequential: deterministic frame processing, no drop under load —
critical for reproducible graph outputs.
