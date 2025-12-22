# STCM Assistive Navigation

The **Semantic Topological Cognitive Mapping (STCM)** workspace packages the perception stack into a ROS 2 Humble friendly layout. It exposes Python nodes that listen to synchronized RGB–D data, run GroundingDINO + MobileSAM, and publish semantic graphs that can be consumed by a new robot for high level reasoning.

## Requirements
- Ubuntu 22.04 with ROS 2 Humble (desktop or ros-base)
- CUDA Toolkit 12.x (tested with 12.4) + cuDNN 9
- Conda or Miniconda for Python environment management
- System dependencies resolved with `rosdep` (`rclpy`, `cv_bridge`, `message_filters`, `tf_transformations`, etc.)

## Installation

**See the main [README.md](../README.md) in the repository root for complete installation instructions.**

Quick summary:

1. Create conda environment: `conda create -n stcm_env python=3.10 -y`
2. Activate environment: `conda activate stcm_env`
3. Install PyTorch with CUDA: `pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121`
4. Install GroundingDINO from source (recommended)
5. Install remaining dependencies: `pip install -r requirements.txt`
6. Build with colcon: `colcon build --packages-select stcm`

Always activate the conda environment before sourcing ROS 2:
```bash
conda activate stcm_env
source /opt/ros/humble/setup.bash
source install/setup.bash
```

## Model checkpoints
The perception modules expect the pretrained weights to live under the workspace `./models` directory (override with `STCM_CKPT_DIR`). The helper CLI downloads and renames everything for you:

```bash
ros2 run stcm stcm_download_checkpoints               # download all defaults into ./models
ros2 run stcm stcm_download_checkpoints --list        # inspect the available models
ros2 run stcm stcm_download_checkpoints --models mobilesam --target /data/ckpts
```

Set `export STCM_CKPT_DIR=/data/ckpts` when using a custom directory. GroundingDINO weights are fetched automatically from the HuggingFace cache; MobileSAM and the optional DepthAnything file are loaded from the checkpoint directory.

## Launching ROS 2 nodes
Use the provided launch file for quick bring-up:

```bash
ros2 launch stcm semantic_mapping.launch.py \
  text_prompt:="table . chair . door ." \
  graph_output_path:="/tmp/semantic_graph.json" \
  run_updater:=false
```

Key parameters (set via the launch file or `ros2 param set`):
- `rgb_topic`, `depth_topic`, `camera_info_topic`: remap to your RGB-D driver topics.
- `camera_frame`, `base_frame`, `world_frame`: match your TF tree (e.g., `camera_link`, `base_link`, `map`).
- Set `use_projected_lidar: true` to ingest the `/lidar_points_projected` cloud instead of the depth image; tweak `projected_lidar_topic` and `projected_lidar_frame` if your fusion node uses different names or frames.
- `target_labels` / `target_label_thresholds`: object classes of interest and per-class merge radius.
- `text_prompt`, `box_threshold`, `text_threshold`: detection prompt and thresholds.
- `graph_output_path` / `graph_input_path`: where semantic graphs are stored.
- `offline_sequential`, `rosbag_path`, `offline_frame_stride`: enable rosbag playback and process every Nth RGB frame offline.

Run the updater separately once you have an initial graph:

```bash
ros2 run stcm semantic_map_updater graph_input_path:=/tmp/semantic_graph.json
```

## Generating semantic graphs on a new robot
1. Ensure TF publishes the camera and base frames into `map`.
2. Calibrate or set the `target_labels` to the objects that matter to your task.
3. Source the workspace and run the builder node (see above). Graph nodes are published as `visualization_msgs/MarkerArray` on `semantic_graph/nodes` for RViz inspection.
4. Persist the generated graph (`graph_output_path`) and feed it back to the updater node to maintain consistency as the robot explores.

## Tests & utilities
The legacy perception demos are still available under `stcm/test`. For example:

```bash
cd stcm
python test/test_gdino_sam.py imgs/irvl-clutter-test.png
python test/test_depth_anything.py imgs/color-000089.png
```

These scripts now import the `stcm.core` modules directly and are useful for quick sanity checks outside of ROS.

## Checkpoint & data directories
- Checkpoints: `./models` (override via `STCM_CKPT_DIR`)
- RViz config: `stcm/config/semantic_mapping.rviz`
- Output graphs: configurable per node (`graph_output_path`)

## License
MIT License © 2025 Anhar Risnumawan. External models (GroundingDINO, MobileSAM, DepthAnything) retain their original licenses—review them before deployment.
