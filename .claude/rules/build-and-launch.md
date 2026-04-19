# Build and Launch Rules

## Build

`ament_python` package. From repo root:

```bash
colcon build --packages-select stcm
# clean rebuild:
rm -rf build/ install/ log/
colcon build --packages-select stcm
```

Entry points in `stcm/setup.py`:
- `semantic_map_builder` — main node, builds semantic graph
- `semantic_map_updater` — updates existing graph
- `stcm_download_checkpoints` — util

Slash command: `/build` or `/clean-build`.

## Launch (primary interface)

File: `stcm/launch/semantic_mapping.launch.py`. YAML config +
CLI override supported.

```bash
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=$(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping_params.yaml

# override specific params:
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=path/to/config.yaml \
  text_prompt:="chair . table . laptop ." \
  graph_output_path:=/tmp/my_graph.json \
  run_updater:=false
```

YAML at `stcm/config/semantic_mapping_params.yaml`. Key fields:
- `text_prompt` — space-separated classes, each ending ` .`
- `graph_output_path` — where builder saves JSON
- `use_sim_time` — `true` for bag/Gazebo, `false` live
- `run_updater` — launch updater alongside builder
- `offline_sequential` — deterministic rosbag2 sequential
- `rosbag_path`, `rosbag_storage_id` — offline replay config
- `groundingdino_checkpoint`, `mobilesam_checkpoint`, `depth_anything_checkpoint`

Slash command: `/launch`.

## Running individual nodes

```bash
ros2 run stcm semantic_map_builder --ros-args \
  -p text_prompt:="table . chair . door ." \
  -p graph_output_path:=/tmp/semantic_graph.json

ros2 run stcm semantic_map_updater --ros-args \
  -p graph_input_path:=/tmp/semantic_graph.json
```

## Key ROS parameters

**Topics + frames:**
- `rgb_topic`, `depth_topic`, `camera_info_topic`
- `camera_frame` (e.g. `head_camera_rgb_optical_frame`)
- `base_frame` (e.g. `base_link`), `world_frame` (e.g. `map`)

**Detection:**
- `target_labels` — tracked classes list
- `target_label_thresholds` — per-class merge radius meters
- `box_threshold`, `text_threshold` — detection confidence (default 0.55)

**Graph:**
- `graph_output_path`, `graph_input_path`
- `processing_period` — frame interval seconds (default 1.0)

**Instance GNG (i-GNG):**
- `gng_enabled`, `gng_per_label`
- `gng_max_nodes`, `gng_lambda`, `gng_max_age`, `gng_eps_w`, `gng_eps_n`,
  `gng_alpha`, `gng_beta`
- `gng_min_observations_to_commit`
- `gng_cluster_merge_distance`, `gng_outlier_gate_meters`

**Place GNG (topological, paper STCM):**
- `place_gng_enabled`, `place_gng_distance_threshold`
- `place_gng_eps_w`, `place_gng_eps_n`, `place_gng_max_edge_age`
- `place_gng_max_nodes`, `place_gng_lambda`, `place_gng_alpha`, `place_gng_beta`
- `place_gng_semantic_alpha`, `place_gng_semantic_aggregation` (`max`/`sum`)
- `place_gng_use_second_best_edge`, `place_gng_use_transition_edges`
- `place_gng_update_when_empty`
- `place_gng_input_path`, `place_gng_output_path` (publishes on
  `semantic_graph/place_graph`)

Place graph nodes/edges separate from object graph. RViz markers use
`world_frame` as fixed frame.

## Adding a new object class

1. Append to `text_prompt`: `"table . chair . door . bookshelf ."`
2. Add to `target_labels`: `["table", "chair", "door", "bookshelf"]`
3. Matching threshold: `target_label_thresholds: [2.0, 0.6, 2.0, 1.5]`
4. Relaunch.

## Changing camera topics for a new robot

1. `ros2 topic list | grep image`
2. Edit YAML or pass params:
   - `rgb_topic:=/my_camera/rgb/image_raw`
   - `depth_topic:=/my_camera/depth/image_raw`
   - `camera_info_topic:=/my_camera/rgb/camera_info`
3. Update frames match TF tree:
   - `camera_frame:=my_camera_optical_frame`
   - `base_frame:=base_footprint`, `world_frame:=map`
