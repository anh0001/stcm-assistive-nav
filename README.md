# STCM Assistive Navigation

![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04-orange?logo=ubuntu&logoColor=white)
![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-blue?logo=ros&logoColor=white)

The **Semantic Topological Cognitive Mapping (STCM)** workspace packages the perception stack into a ROS 2 Humble friendly layout. It exposes Python nodes that listen to synchronized RGB–D data, run GroundingDINO + MobileSAM, and publish semantic graphs that can be consumed by a new robot for high level reasoning.

**Quick Links:**
- [Installation](#installation) - Detailed setup instructions
- [Model Checkpoints](#model-checkpoints) - Download pretrained weights
- [Launching ROS 2 Nodes](#launching-ros-2-nodes) - Running the system

## Requirements
- Ubuntu 22.04 with ROS 2 Humble (desktop or ros-base)
- CUDA Toolkit 12.x (tested with 12.4) + cuDNN 9
- System Python 3.10 (default on Ubuntu 22.04). Packages are installed via `python3 -m pip --user` into an isolated `PYTHONUSERBASE`.
- System dependencies resolved with `rosdep` (`rclpy`, `cv_bridge`, `message_filters`, `tf_transformations`, etc.)

## Installation

### Quick Start (Automated)

For automated installation, use the provided setup script:

```bash
cd ~/stcm-assistive-nav
./setup_stcm_env.sh
```

This script will:
1. Export (or create) `PYTHONUSERBASE=${HOME}/.local/stcm_sys_py310`
2. Upgrade pip + install PyTorch 2.4.0 (CUDA 12.1 wheels), GroundingDINO, MobileSAM, and the rest of the Python dependencies into that isolated user base
3. Install ROS dependencies via `rosdep`
4. Build the ROS 2 package and verify imports

### Manual Installation

If you prefer to install manually or need to troubleshoot, follow these steps. All commands rely on the system `python3` and an isolated user base so ROS continues using the distro interpreter.

### System Compatibility
Your setup: **Ubuntu 22.04 + ROS 2 Humble + CUDA toolkit 12.4 + cuDNN 9**

PyTorch wheels ship their own CUDA runtime (cu121) + cuDNN, so your system CUDA 12.4 installation is fine—the NVIDIA driver just needs to meet the minimum version required by the wheel.

### Step 0: Export an isolated `PYTHONUSERBASE`

```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"  # pick any clean directory
mkdir -p "$PYTHONUSERBASE"
echo 'export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"' >> ~/.bashrc  # optional helper
```

All subsequent `python3 -m pip install --user ...` commands will land inside that directory. This keeps ROS’ `/usr/bin/python3` happy without polluting `~/.local`.

### Step 1: Upgrade pip inside that user base

```bash
python3 -m pip install --upgrade --user pip
```

### Step 2: Install CUDA-enabled PyTorch

```bash
python3 -m pip install --user \
  torch==2.4.0+cu121 \
  torchvision==0.19.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python3 - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda runtime:', torch.version.cuda)
PY
```

### Step 3: Install GroundingDINO

```bash
cd ~
git clone https://github.com/IDEA-Research/GroundingDINO.git
python3 -m pip install --user --no-build-isolation -e ./GroundingDINO
```

> Prefer the source install so the CUDA extensions compile against the PyTorch wheel above. A `pip install groundingdino-py` fallback works, but the source tree is more flexible for patches.

### Step 4: Install MobileSAM

```bash
python3 -m pip install --user git+https://github.com/ChaoningZhang/MobileSAM.git
```

### Step 5: Install remaining requirements

```bash
cd ~/stcm-assistive-nav/stcm
python3 -m pip install --user -r requirements.txt
```

### Step 6: Install ROS 2 dependencies

```bash
cd ~/stcm-assistive-nav
source /opt/ros/humble/setup.bash
rosdep install --from-paths stcm --ignore-src -y
```

### Step 7: Build the ROS 2 package

```bash
cd ~/stcm-assistive-nav
colcon build --packages-select stcm
```

### Step 8: Setup your shell environment

For every new terminal session:

```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"  # ensure this matches the path you picked earlier
source /opt/ros/humble/setup.bash
source ./install/setup.bash
```

## Model checkpoints
The perception modules expect the pretrained weights to live under `~/.stcm/ckpts` (override with `STCM_CKPT_DIR`). The helper CLI downloads and renames everything for you:

```bash
ros2 run stcm stcm_download_checkpoints --target ./models # download all defaults into ./models folder
ros2 run stcm stcm_download_checkpoints --list        # inspect the available models
ros2 run stcm stcm_download_checkpoints --models mobilesam --target /data/ckpts
```

Set `export STCM_CKPT_DIR=/data/ckpts` when using a custom directory. GroundingDINO weights are fetched automatically from the HuggingFace cache; MobileSAM and the optional DepthAnything file are loaded from the checkpoint directory.

## Required Inputs

**Camera Topics** (find with `ros2 topic list | grep image`):
- RGB image (e.g., `/camera/color/image_raw`)
- Depth image (e.g., `/camera/aligned_depth_to_color/image_raw`)
- Camera info (e.g., `/camera/color/camera_info`)
- Optional: enable `use_projected_lidar: true` plus `projected_lidar_topic` to use `/lidar_points_projected` instead of `depth_topic`. The PointCloud2 must include `u`/`v` pixel fields and XYZ in the lidar frame so SAM masks can pick the matching points; override `projected_lidar_frame` when the header lacks the desired frame. When `use_projected_lidar: false`, the node uses `depth_topic` (including Depth Anything outputs).

**TF Transforms** (verify with `ros2 run tf2_tools view_frames`):
- `camera_frame → base_frame → world_frame` (e.g., `camera_optical_frame → base_link → map`)
- The `camera_frame` parameter is also used to override the RGB topic's `frame_id` when the nodes republish segmented images, so use it to correct cameras that advertise the wrong frame.
- Without SLAM/localization, publish static transform: `ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link`
- When replaying rosbag data that loops or resets time, keep `reset_tf_on_time_jump: true` in the YAML to clear stale TF history.

**Object Classes** (configure in YAML):
```yaml
text_prompt: "table . chair . door ."  # each class ends with " ."
target_labels: ["table", "chair", "door"]
target_label_thresholds: [2.0, 0.6, 2.0]  # merge radius in meters
```

**Detection Filtering** (optional, configure in YAML):
```yaml
filter_conf_bound: 1.0
filter_y_val: 1.0
filter_percent_width: 0.9
filter_percent_height: 0.9
filter_percent_area: 0.005
filter_enabled: true
```

**Instance Management (i-GNG)** (optional, configure in YAML):
```yaml
gng_enabled: true
gng_per_label: true
gng_max_nodes: 1000
gng_lambda: 200
gng_max_age: 200
gng_eps_w: 0.05
gng_eps_n: 0.0006
gng_alpha: 0.95
gng_beta: 0.9995
gng_min_observations_to_commit: 3
gng_cluster_merge_distance: 0.5
gng_outlier_gate_meters: 0.0
```

**Topological Place GNG (paper STCM)** (optional, configure in YAML):
Builds a place graph from the robot trajectory (2D map-frame pose), adapts node prototypes online,
adds edges on transitions, and fuses detection confidences into node-level semantic scores/labels.
Outputs a MarkerArray on `semantic_graph/place_graph` and embeds the place graph in the STCM JSON
(`graph_output_path`). Set `place_gng_output_path` to a different file if you need a standalone
place graph export; otherwise it defaults to `graph_output_path`.
```yaml
place_gng_enabled: true
place_gng_distance_threshold: 1.5  # D_new (meters)
place_gng_eps_w: 0.1               # winner learning rate
place_gng_eps_n: 0.01              # neighbor learning rate
place_gng_max_edge_age: 50         # a_max
place_gng_semantic_alpha: 0.1      # semantic fusion rate
place_gng_semantic_aggregation: "max"  # "max" or "sum"
place_gng_use_second_best_edge: true
place_gng_use_transition_edges: true
place_gng_update_when_empty: false
```

**Spatial Relationships** (optional, configure in YAML):
```yaml
edge_distance_threshold: 3.0  # max distance (meters) to connect objects in the graph
```

## Launching ROS 2 nodes
Edit `stcm/config/semantic_mapping_params.yaml` to configure topics, TF frames, object classes, and checkpoints. Then launch:

```bash
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=$(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping_params.yaml
```

You can still override individual values at launch time (`text_prompt:=...`, `graph_output_path:=...`,
`use_sim_time:=...`, `run_updater:=...`), but keeping them in the YAML makes repeated runs easier.

All parameters are configured in the YAML file (topics, TF frames, detection settings, output paths).

For deterministic offline runs (single-thread sequential rosbag reader), enable:
```yaml
offline_sequential: true
rosbag_path: "/path/to/rosbag2_dir"
```
Or override at launch time:
```bash
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=$(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping_params.yaml \
  offline_sequential:=true \
  rosbag_path:=/path/to/rosbag2_dir
```

## Semantic Graph Simulator (RViz)

Use the STCM JSON to drive a 2D RViz simulation with the language planner.

```bash
colcon build --packages-select stcm_planner
source install/setup.bash
ros2 run stcm_planner semantic_graph_simulator --ros-args -p graph_path:=/tmp/stcm.json
```

In RViz, add MarkerArray for `/semantic_graph_sim/nodes` and Marker displays for
`/semantic_graph_sim/path` and `/semantic_graph_sim/robot`. Send queries with:

```bash
ros2 run stcm_planner language_query_publisher
```

The query publisher sends commands on `/stcm_planner_query`.

Run the updater separately once you have an initial graph:

```bash
ros2 run stcm semantic_map_updater graph_input_path:=/tmp/stcm.json
```

## Generating semantic graphs on a new robot
1. Ensure TF publishes the camera and base frames into `map`.
2. Calibrate or set the `target_labels` to the objects that matter to your task.
3. Source the workspace and run the builder node (see above). Graph nodes are published as `visualization_msgs/MarkerArray` on `semantic_graph/nodes` for RViz inspection.
4. Persist the generated graph (`graph_output_path`) and feed it back to the updater node to maintain consistency as the robot explores.

## Tests & utilities
The legacy perception demos are still available under `stcm/test`.

**Important**: Before running tests, ensure your environment is properly set up:
```bash
# 1. Export PYTHONUSERBASE so python finds the isolated deps
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"

# 2. Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# 3. Source your workspace (run from repository root)
source install/setup.bash
```

Then run tests from the **repository root**:
```bash
python stcm/test/test_gdino_sam.py stcm/imgs/irvl-clutter-test.png
python stcm/test/test_depth_anything.py stcm/imgs/color-000089.png
```

To retroactively add spatial edges to an existing graph:
```bash
python3 stcm/tools/add_edges_to_graph.py stcm.json --output stcm_with_edges.json --distance 3.0
```

These scripts import the `stcm.core` modules directly and are useful for quick sanity checks outside of ROS.

## Checkpoint & data directories
- Checkpoints: `~/.stcm/ckpts` (override via `STCM_CKPT_DIR`)
- RViz config: `stcm/config/semantic_mapping.rviz`
- Output graphs: configurable per node (`graph_output_path`, default `output/stcm.json`, includes `semantic_graph`, `place_graph`, and `llm` summary)

## License
MIT License © 2025 Anhar Risnumawan. External models (GroundingDINO, MobileSAM, DepthAnything) retain their original licenses—review them before deployment.
