# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**STCM Assistive Navigation** is a ROS 2 Humble package that implements Semantic Topological Cognitive Mapping for robotic perception. It combines GroundingDINO (open-vocabulary object detection) with MobileSAM (efficient segmentation) to build semantic graphs from RGB-D data streams. These graphs represent spatial relationships between detected objects and can be used for high-level robot reasoning and navigation.

The system processes synchronized RGB-D camera streams, detects objects using text prompts, segments them, calculates their 3D positions in the world frame, and maintains a NetworkX graph that tracks objects and their spatial relationships over time.

## Environment Setup

ROS nodes must run on `/usr/bin/python3` (keep the `#!/usr/bin/python3` shebang) to match ROS 2’s assumptions, so skip conda and prime a dedicated user base for this project:

```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"  # add to your shell profile / VS Code tasks
python3 -m pip install --upgrade --user pip
python3 -m pip install --user ros2-numpy
python3 -m pip install --user torch torchvision torchaudio
# install any additional imports the nodes require
```

Follow PyTorch’s selector to pick the correct CUDA wheels; their binaries already ship the CUDA runtime, so the NVIDIA driver version matters more than your local toolkit. Once the user base is ready, source ROS and the workspace (after ensuring `PYTHONUSERBASE` is still exported):

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

This keeps the runtime ROS-friendly while isolating ML dependencies in `$HOME/.local/stcm_sys_py310`.

## Build System

```bash
# Build the package (from repository root)
colcon build --packages-select stcm

# Clean build
rm -rf build/ install/ log/
colcon build --packages-select stcm
```

The package uses `ament_python` build type. Entry points are defined in [stcm/setup.py](stcm/setup.py):
- `semantic_map_builder` - Main node for building semantic graphs
- `semantic_map_updater` - Node for updating existing graphs
- `stcm_download_checkpoints` - Utility for downloading model weights

## Model Checkpoints

Model weights are stored in `./models` by default (override with `STCM_CKPT_DIR` environment variable):

```bash
# Download all default checkpoints to ./models
ros2 run stcm stcm_download_checkpoints

# List available models
ros2 run stcm stcm_download_checkpoints --list

# Download specific model to custom location
ros2 run stcm stcm_download_checkpoints --models mobilesam --target /data/ckpts
export STCM_CKPT_DIR=/data/ckpts
```

Expected checkpoint structure:
- `models/gdino/groundingdino_swint_ogc.pth` - GroundingDINO weights
- `models/mobilesam/vit_t.pth` - MobileSAM weights
- `models/depth_anything/depth_anything_vitb14.pth` - Optional depth model

## Running the System

### Launch File Usage

The primary interface is [stcm/launch/semantic_mapping.launch.py](stcm/launch/semantic_mapping.launch.py), which supports configuration via YAML and command-line overrides:

```bash
# Using config file (recommended for repeated runs)
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=$(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping_params.yaml

# Override specific parameters at launch time
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=path/to/config.yaml \
  text_prompt:="chair . table . laptop ." \
  graph_output_path:=/tmp/my_graph.json \
  run_updater:=false
```

Edit [stcm/config/semantic_mapping_params.yaml](stcm/config/semantic_mapping_params.yaml) to configure:
- `text_prompt` - Space-separated object classes, each ending with ` .` (e.g., `"table . chair . door ."`)
- `graph_output_path` - Where to save the semantic graph JSON
- `use_sim_time` - Set to `true` for bag playback or Gazebo, `false` for live robot
- `run_updater` - Launch the updater node alongside the builder
- `offline_sequential` - Enable deterministic rosbag2 sequential processing
- `rosbag_path` - Path to the rosbag2 directory (required when offline)
- `rosbag_storage_id` - Storage plugin for rosbag2 (default `sqlite3`)
- `groundingdino_checkpoint`, `mobilesam_checkpoint`, `depth_anything_checkpoint` - Paths to model weights

### Running Individual Nodes

```bash
# Builder node (creates new graph)
ros2 run stcm semantic_map_builder \
  --ros-args \
  -p text_prompt:="table . chair . door ." \
  -p graph_output_path:=/tmp/semantic_graph.json

# Updater node (maintains existing graph)
ros2 run stcm semantic_map_updater \
  --ros-args \
  -p graph_input_path:=/tmp/semantic_graph.json
```

### Key ROS Parameters

Parameters can be set in the config YAML or passed via `--ros-args -p`:

**Topics and frames:**
- `rgb_topic`, `depth_topic`, `camera_info_topic` - Input RGB-D stream topics
- `camera_frame`, `base_frame`, `world_frame` - TF frames (e.g., `head_camera_rgb_optical_frame`, `base_link`, `map`)

**Detection configuration:**
- `target_labels` - List of object classes to track (e.g., `["table", "chair", "door"]`)
- `target_label_thresholds` - Per-class merge radius in meters (e.g., `[2.0, 2.0, 0.6]`)
- `box_threshold`, `text_threshold` - Detection confidence thresholds (default: 0.55)

**Graph management:**
- `graph_output_path` - Path where builder saves the graph
- `graph_input_path` - Path where updater reads the graph
- `processing_period` - Frame processing interval in seconds (default: 1.0)

**Instance management (i-GNG):**
- `gng_enabled` - Enable per-label Growing Neural Gas instance management
- `gng_per_label` - Maintain a separate GNG model per label (recommended)
- `gng_max_nodes`, `gng_lambda`, `gng_max_age`, `gng_eps_w`, `gng_eps_n`, `gng_alpha`, `gng_beta` - GNG tuning knobs
- `gng_min_observations_to_commit` - Observations required before adding a graph node
- `gng_cluster_merge_distance` - Merge/associate clusters within this radius (meters)
- `gng_outlier_gate_meters` - Ignore samples farther than this from existing instances (meters)

**Topological place GNG (paper STCM):**
- `place_gng_enabled` - Enable place-graph learning from the robot pose stream
- `place_gng_distance_threshold` - Pose insertion threshold for i-GNG (meters)
- `place_gng_eps_w`, `place_gng_eps_n` - Winner/neighbor learning rates
- `place_gng_max_edge_age` - Edge age threshold for pruning
- `place_gng_max_nodes`, `place_gng_lambda`, `place_gng_alpha`, `place_gng_beta` - i-GNG tuning knobs
- `place_gng_semantic_alpha` - Semantic score fusion rate
- `place_gng_semantic_aggregation` - Aggregate class evidence via `max` or `sum`
- `place_gng_use_second_best_edge` / `place_gng_use_transition_edges` - Edge creation toggles
- `place_gng_update_when_empty` - Whether to update scores when no detections are present
- `place_gng_input_path` / `place_gng_output_path` - Place graph JSON paths (output published on `semantic_graph/place_graph`)
  - Place graph nodes/edges are separate from the object graph; RViz markers use the `world_frame` fixed frame.

## Testing

Tests are located in [stcm/test/](stcm/test/) and test individual perception components outside of ROS:

```bash
# Ensure environment is set up (from repository root)
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"  # required so pip-installed deps resolve
source /opt/ros/humble/setup.bash
source install/setup.bash

# Run GroundingDINO + MobileSAM test
python stcm/test/test_gdino_sam.py stcm/imgs/irvl-clutter-test.png

# Run depth estimation test
python stcm/test/test_depth_anything.py stcm/imgs/color-000089.png

# Run on custom image
python stcm/test/test_gdino_sam.py /path/to/image.png

# Test with ROS image publisher (publishes test images to ROS topics)
python stcm/test/ros_test_images.py
```

The test scripts import `stcm.core` modules directly and use the helper [stcm/test/_ckpt.py](stcm/test/_ckpt.py) to ensure the workspace checkpoint directory is used. Tests output annotated images showing detections and segmentations.

## Architecture

### Package Structure

```
stcm/
├── stcm/                          # Main Python package
│   ├── core/                      # Core perception modules
│   │   ├── perception.py          # GroundingDINOObjectPredictor, SegmentAnythingPredictor, DepthPredictor
│   │   ├── vision_utils.py        # Detection filtering, annotation, mask utilities
│   │   ├── checkpoints.py         # Checkpoint path management
│   │   ├── datasets/              # Dataset loaders (OCID, OSD for evaluation)
│   │   └── cfg/                   # GroundingDINO config files
│   ├── nodes/                     # ROS 2 nodes
│   │   ├── semantic_map_builder.py   # Main builder node
│   │   └── semantic_map_updater.py   # Graph updater node
│   ├── tools/                     # Utilities
│   │   └── checkpoint_manager.py  # CLI for downloading checkpoints
│   ├── image_listener.py          # RGB-D synchronization and TF tracking
│   ├── map_utils.py               # Graph operations, spatial queries, JSON serialization
│   └── ros_utils.py               # ROS message conversion utilities
├── test/                          # Standalone perception tests
├── config/                        # Launch configuration files
├── launch/                        # ROS 2 launch files
└── imgs/                          # Test images
```

### Core Components

**Perception Pipeline ([stcm/stcm/core/perception.py](stcm/stcm/core/perception.py)):**
- `GroundingDINOObjectPredictor` - Open-vocabulary object detection using text prompts
- `SegmentAnythingPredictor` - Instance segmentation using MobileSAM with box prompts from GDINO
- `DepthAnythingPredictor` - Optional monocular depth estimation (not used in RGB-D mode)

All predictors inherit from `CommonContextObject` which provides automatic CUDA device management and logging.

**Image Synchronization ([stcm/stcm/image_listener.py](stcm/stcm/image_listener.py)):**
- `ImageListener` class synchronizes RGB and depth topics using `message_filters.ApproximateTimeSynchronizer`
- Maintains TF buffer for camera → base → world transforms
- Blocks until camera intrinsics are received from `CameraInfo` topic
- Thread-safe access to latest synchronized frame via `.im`, `.depth`, `.intrinsics`

**Semantic Graph Management ([stcm/stcm/map_utils.py](stcm/stcm/map_utils.py)):**
- Graph is a NetworkX undirected graph stored as JSON
- Nodes represent detected objects with attributes: `label`, `pose` (3D position in world frame), `count` (detection frequency)
- `is_nearby_in_map()` checks if a new detection is close to existing graph nodes (uses per-class thresholds)
- `pose_in_map_frame()` transforms detected object positions from camera to world frame using TF and depth
- `save_graph_json()` / `load_graph_json()` handle persistence

**Builder Node ([stcm/stcm/nodes/semantic_map_builder.py](stcm/stcm/nodes/semantic_map_builder.py)):**
1. Receives synchronized RGB-D frames via `ImageListener`
2. Runs GroundingDINO detection with text prompt → bounding boxes + labels
3. Runs MobileSAM on each box → instance masks
4. For each detection:
   - Calculate 3D centroid from mask and depth
   - Transform to world frame using TF
   - Check if nearby existing graph node (using `target_label_thresholds`)
   - If new: add node; if existing: increment count
5. Publish graph as `visualization_msgs/MarkerArray` on `semantic_graph/nodes` for RViz
6. Periodically save graph to `graph_output_path`

**Updater Node ([stcm/stcm/nodes/semantic_map_updater.py](stcm/stcm/nodes/semantic_map_updater.py)):**
- Loads existing graph from `graph_input_path`
- Continuously updates and republishes as new detections arrive
- Maintains consistency as robot explores and revisits locations

### Detection and Merging Logic

The system uses spatial proximity to merge repeated detections of the same object:

1. `target_labels` defines which classes to track (e.g., `["table", "chair", "door"]`)
2. `target_label_thresholds` defines per-class merge radius in meters (e.g., `[2.0, 2.0, 0.6]`)
   - Large objects like tables use larger thresholds (2.0m)
   - Small objects like chairs use smaller thresholds (0.6m)
3. When a new detection occurs, `is_nearby_in_map()` checks Euclidean distance to all existing nodes with the same label
4. If distance < threshold: update existing node (increment count), else: create new node

## Common Workflows

### Adding a New Object Class

1. Edit the text prompt to include the new class: `text_prompt: "table . chair . door . bookshelf ."`
2. Add to target labels: `target_labels: ["table", "chair", "door", "bookshelf"]`
3. Add corresponding threshold: `target_label_thresholds: [2.0, 0.6, 2.0, 1.5]`
4. Relaunch the system

### Changing Camera Topics for a New Robot

1. Find your robot's RGB-D topic names: `ros2 topic list | grep image`
2. Edit config YAML or pass parameters:
   - `rgb_topic:=/my_camera/rgb/image_raw`
   - `depth_topic:=/my_camera/depth/image_raw`
   - `camera_info_topic:=/my_camera/rgb/camera_info`
3. Update frame names to match your TF tree:
   - `camera_frame:=my_camera_optical_frame`
   - `base_frame:=base_footprint`
   - `world_frame:=map`

### Debugging Detection Issues

1. Check published detection visualization: `ros2 topic echo /semantic_graph/segmented_image`
2. Check graph markers in RViz: Add `MarkerArray` display for `/semantic_graph/nodes`
3. Adjust detection thresholds if too few/many detections:
   - Lower `box_threshold` and `text_threshold` for more detections (default: 0.55)
   - Raise thresholds for fewer, more confident detections
4. Run standalone test to isolate ROS issues: `python stcm/test/test_gdino_sam.py image.png`

### Visualizing the Graph in RViz

1. Launch RViz: `rviz2 -d $(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping.rviz`
2. Add MarkerArray display, topic: `/semantic_graph/nodes`
3. Set fixed frame to `map` or your world frame
4. Graph nodes appear as colored spheres with labels

## Dependencies

**Python packages (install via pip into `$PYTHONUSERBASE`):**
- PyTorch 2.4.0 + CUDA 12.1 (installed from torch index)
- GroundingDINO (installed from source via `pip install -e` or `groundingdino-py`)
- MobileSAM (installed from GitHub)
- HuggingFace `transformers`, `huggingface-hub` (for model loading)
- Vision: `opencv-python`, `Pillow`, `supervision`, `open3d`, `scikit-image`
- Graph: `networkx`
- ROS bridge: `ros2-numpy`

**ROS 2 packages (installed via rosdep):**
- Standard: `rclpy`, `sensor_msgs`, `geometry_msgs`, `visualization_msgs`
- TF: `tf2_ros`, `tf_transformations`
- Sensors: `cv_bridge`, `message_filters`

**System requirements:**
- Ubuntu 22.04 + ROS 2 Humble
- CUDA 12.x + cuDNN 9 (for GPU acceleration)

## Important Notes

- **Environment prep matters**: Export `PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"` (or your chosen path) before sourcing `/opt/ros/humble/setup.bash` so the system interpreter sees the isolated dependencies
- **Test scripts must be run from repository root**: They use relative paths to `stcm/test/` and `stcm/imgs/`
- **Text prompt format**: Each object class must end with ` .` (space + period) for GroundingDINO
- **Checkpoint directory**: Override default `./models` with `export STCM_CKPT_DIR=/custom/path`
- **TF requirement**: The builder node requires TF transforms from camera → world frame to exist before processing frames
- **Graph persistence**: Graphs are saved as JSON with NetworkX node-link format (readable and editable)

## System Python Runtime Checklist

- Keep ROS and perception processes on `/usr/bin/python3` with the dedicated `PYTHONUSERBASE` (`$HOME/.local/stcm_sys_py310`). Leave node shebangs untouched.
- Make sure `PYTHONUSERBASE` is exported before sourcing ROS or running VS Code tasks/launchers so pip-installed packages resolve. Bake it into your shell profile or wrapper scripts.
- Install every new dependency with `python3 -m pip install --user ...` while the user base is active; never add more packages to the global `~/.local`.
- If you still split perception into a helper service, start it from the same interpreter and document IPC parameters such as `perception_endpoint`, `ipc_transport`, and `request_timeout` in YAML + launch files.
- Pros: ROS-friendly, minimal ABI fights. Cons: you must maintain ML deps in system-python land—but they stay isolated inside `PYTHONUSERBASE`.
