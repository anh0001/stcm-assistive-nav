# Perception Pipeline Rules

## Core components

**Perception (`stcm/stcm/core/perception.py`):**
- `GroundingDINOObjectPredictor` — open-vocab detection via text prompts
- `SegmentAnythingPredictor` — MobileSAM instance seg w/ box prompts from GDINO
- `DepthAnythingPredictor` — optional mono depth (not used in RGB-D mode)

All inherit `CommonContextObject` → auto CUDA device mgmt + logging.

**Image sync (`stcm/stcm/image_listener.py`):**
- `ImageListener` syncs RGB + depth via `message_filters.ApproximateTimeSynchronizer`
- Holds TF buffer for camera → base → world
- Blocks until `CameraInfo` received
- Thread-safe access to latest frame via `.im`, `.depth`, `.intrinsics`

**Semantic graph (`stcm/stcm/map_utils.py`):**
- NetworkX undirected graph, JSON persisted
- Node attrs: `label`, `pose` (3D world), `count` (detection freq)
- `is_nearby_in_map()` — check new detection close to existing nodes (per-class thresholds)
- `pose_in_map_frame()` — transform cam → world via TF + depth
- `save_graph_json()` / `load_graph_json()` — NetworkX node-link format

## Builder node (`stcm/stcm/nodes/semantic_map_builder.py`)

1. Receives synced RGB-D frames via `ImageListener`
2. GroundingDINO detect w/ text prompt → boxes + labels
3. MobileSAM on each box → instance masks
4. Per detection:
   - Compute 3D centroid (mask + depth)
   - Transform cam → world via TF
   - Check near existing (per-class threshold)
   - New → add node. Existing → increment count.
5. Publishes `visualization_msgs/MarkerArray` on `semantic_graph/nodes` for RViz
6. Periodic save to `graph_output_path`

## Updater node (`stcm/stcm/nodes/semantic_map_updater.py`)

- Loads graph from `graph_input_path`
- Continuous update + republish as new detections arrive
- Maintains consistency as robot explores + revisits

## Detection + merging logic

Spatial proximity merges repeat detections of same object:

1. `target_labels` defines tracked classes
2. `target_label_thresholds` — per-class merge radius (meters)
   - Big objects (table) → big threshold (2.0m)
   - Small objects (chair) → small threshold (0.6m)
3. On detection, `is_nearby_in_map()` checks Euclidean dist to all existing
   nodes w/ same label
4. Dist < threshold → update existing (increment count).
   Else → create new node.

## Package structure (quick map)

```
stcm/stcm/
├── core/                  # Perception modules
│   ├── perception.py      # GDINO, MobileSAM, Depth predictors
│   ├── vision_utils.py    # Detection filtering, annotation, mask utils
│   ├── checkpoints.py     # Checkpoint path mgmt
│   ├── datasets/          # OCID, OSD loaders for eval
│   └── cfg/               # GroundingDINO configs
├── nodes/                 # ROS 2 nodes
│   ├── semantic_map_builder.py
│   └── semantic_map_updater.py
├── tools/
│   └── checkpoint_manager.py
├── image_listener.py      # RGB-D sync + TF tracking
├── map_utils.py           # Graph ops, spatial queries, JSON serialize
└── ros_utils.py           # ROS message conversion

stcm/test/                 # Standalone perception tests
stcm/config/               # Launch YAML
stcm/launch/               # ROS 2 launch files
stcm/imgs/                 # Test images
```
