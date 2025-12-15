# STCM Assistive Navigation — AI Agent Guide

- **What this repo is**: ROS 2 Humble package that turns RGB-D streams into semantic graphs using GroundingDINO + MobileSAM (optionally DepthAnything). Runtime nodes live in [stcm/stcm/nodes/semantic_map_builder.py](stcm/stcm/nodes/semantic_map_builder.py) and [stcm/stcm/nodes/semantic_map_updater.py](stcm/stcm/nodes/semantic_map_updater.py).

- **Runtime topology**: `semantic_map_builder` subscribes to RGB, depth, and CameraInfo, pulls TF for camera→base→map via [stcm/stcm/image_listener.py](stcm/stcm/image_listener.py), runs detection + segmentation, then writes and publishes a semantic graph (markers on `semantic_graph/nodes`, annotated image on `semantic_graph/segmented_image`). `semantic_map_updater` loads an existing graph, removes stale nodes in current FoV, and adds new ones; optional pause topic toggles updates.

- **Launch + params**: The launch entry is [stcm/launch/semantic_mapping.launch.py](stcm/launch/semantic_mapping.launch.py) with YAML defaults in [stcm/config/semantic_mapping_params.yaml](stcm/config/semantic_mapping_params.yaml). CLI overrides: `text_prompt`, `graph_output_path`, `use_sim_time`, `run_updater`, and checkpoint paths. Builder/updater both declare parameters for topics (`rgb_topic`, `depth_topic`, `camera_info_topic`), frames (`camera_frame`, `base_frame`, `world_frame`), thresholds, and checkpoint overrides.

- **Checkpoints**: Expected under `./models` (or `STCM_CKPT_DIR`). Defaults: `models/gdino/groundingdino_swint_ogc.pth`, `models/mobilesam/vit_t.pth`, optional `models/depth_anything/depth_anything_vitb14.pth`. Download via `ros2 run stcm stcm_download_checkpoints` (see [stcm/README.md](stcm/README.md)). Both nodes accept per-run overrides.

- **Perception stack internals**: Detection/segmentation utilities in [stcm/stcm/core/perception.py](stcm/stcm/core/perception.py); masks + visualization helpers in [stcm/stcm/core/vision_utils.py](stcm/stcm/core/vision_utils.py). Builder/updater apply additional filtering (`filter`, `filter_large_boxes`) before graph updates. Graph I/O helpers live in [stcm/stcm/map_utils.py](stcm/stcm/map_utils.py).

- **Semantic graph behavior**: Builder creates nodes when detections are sufficiently far from previous sightings (per-label distance thresholds). Updater removes nodes in-view that are no longer detected and prunes legacy nodes when no detections occur in the current FoV. Graphs are persisted on node destruction to the configured path.

- **Mandatory environment order**: Always `conda activate stcm_env` → `source /opt/ros/humble/setup.bash` → `source ./install/setup.bash`. Skipping ROS setup causes TF lookups and message_filters sync to fail.

- **Build workflow**: From repo root, `colcon build --packages-select stcm` after dependencies are installed (`rosdep install --from-paths stcm --ignore-src -y`). Binary artifacts live in `build/` and `install/`; source edits happen under `stcm/stcm/**`.

- **Tests / quick sanity checks**: Run from repo root with the environment sourced: `python stcm/test/test_gdino_sam.py stcm/imgs/irvl-clutter-test.png` and `python stcm/test/test_depth_anything.py stcm/imgs/color-000089.png`. These import the core predictors directly (no ROS graph needed) but still require checkpoints.

- **Launch usage example**:
  - `ros2 launch stcm semantic_mapping.launch.py config_file:=$(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping_params.yaml`
  - To update an existing graph only: `ros2 run stcm semantic_map_updater graph_input_path:=/tmp/semantic_graph.json`

- **Topic/frame assumptions**: Defaults expect depth in meters (`16UC1` or `32FC1`), TF frames `camera_frame`→`base_frame`→`world_frame` to be available, and synchronized RGB/depth via ApproximateTimeSynchronizer. Remap or adjust thresholds when integrating new sensors.

- **Common pitfalls**: missing checkpoints (MobileSAM lookup fails fast), wrong frame IDs (TF lookup warnings), unsourced ROS env (no parameters/TF), and large boxes being filtered by `filter_large_boxes` if they cover >50% of the image.

- **Style conventions**: Code is Python 3.10, ROS 2 nodes use rclpy Node classes with parameter declarations and publishers. Keep new node parameters aligned with existing names (snake_case, declared in `__init__`). Prefer adding config toggles to the YAML + launch file for new runtime options.

- **Data locations**: RViz config in [stcm/config/semantic_mapping.rviz](stcm/config/semantic_mapping.rviz); sample images in [stcm/imgs](stcm/imgs); persisted graphs default to repo root unless overridden.

- **Licensing**: MIT for this repo; external models maintain their own licenses—do not bundle proprietary weights.