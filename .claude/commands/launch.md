---
description: Launch semantic_mapping with default YAML config, accept overrides
argument-hint: [key:=value ...]
---

Launch STCM semantic mapping pipeline.

Default invocation:
```bash
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=$(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping_params.yaml
```

If user passes extra `key:=value` args via `$ARGUMENTS`, append them.

Common overrides:
- `text_prompt:="chair . table ."`
- `graph_output_path:=/tmp/my_graph.json`
- `run_updater:=false`
- `offline_sequential:=true rosbag_path:=/path/to/bag`

Pre-checks before launching:
1. `install/setup.bash` sourced (otherwise `ros2 launch stcm` fails)
2. Checkpoints exist in `./models/` (or `$STCM_CKPT_DIR`)
3. If live robot: RGB/depth topics publishing (`ros2 topic list | grep image`)
