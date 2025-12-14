# STCM Assistive Navigation

The **Semantic Topological Cognitive Mapping (STCM)** workspace packages the perception stack into a ROS 2 Humble friendly layout. It exposes Python nodes that listen to synchronized RGB–D data, run GroundingDINO + MobileSAM, and publish semantic graphs that can be consumed by a new robot for high level reasoning.

**Quick Links:**
- [Installation](#installation) - Detailed setup instructions
- [Model Checkpoints](#model-checkpoints) - Download pretrained weights
- [Launching ROS 2 Nodes](#launching-ros-2-nodes) - Running the system

## Requirements
- Ubuntu 22.04 with ROS 2 Humble (desktop or ros-base)
- CUDA Toolkit 12.x (tested with 12.4) + cuDNN 9
- Conda or Miniconda for Python environment management
- System dependencies resolved with `rosdep` (`rclpy`, `cv_bridge`, `message_filters`, `tf_transformations`, etc.)

## Installation

### Quick Start (Automated)

For automated installation, use the provided setup script:

```bash
cd ~/stcm-assistive-nav
./setup_stcm_env.sh
```

This script will:
1. Create the `stcm_env` conda environment
2. Install PyTorch 2.4.0 with CUDA 12.1
3. Install GroundingDINO from source
4. Install MobileSAM from GitHub
5. Install all Python dependencies
6. Install ROS 2 dependencies
7. Build the ROS 2 package

**Troubleshooting**: If you see warnings about "Unexpected error writing token file", you can safely ignore them (it's a conda analytics issue) or fix by running:
```bash
conda config --set allow_conda_downgrades true
```

### Manual Installation

If you prefer to install manually or need to troubleshoot, follow these steps:

### System Compatibility
Your setup: **Ubuntu 22.04 + ROS 2 Humble + CUDA toolkit 12.4 + cuDNN 9**

PyTorch wheels ship their own CUDA runtime (cu121), so your system CUDA 12.4 won't conflict. The binary wheels come with their own cuDNN, so your system cuDNN 9 install is not used by PyTorch.

### Step 1: Create Conda Environment

**Option A:** Create from environment file

```bash
conda env create -f environment.yml
conda activate stcm_env
```

**Option B:** Create manually

```bash
# Create dedicated environment with Python 3.10
conda create -n stcm_env python=3.10 -y

# Activate environment
conda activate stcm_env
```

### Step 2: Install CUDA-enabled PyTorch

Install PyTorch 2.4.0 with CUDA 12.1 support (stable, known-good combo):

```bash
pip install \
  torch==2.4.0+cu121 \
  torchvision==0.19.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda version:', torch.version.cuda)"
```

Expected output:
- `torch: 2.4.0+cu121`
- `cuda available: True`
- `cuda version: 12.1` (wheel's internal runtime, not your system 12.4)

### Step 3: Install GroundingDINO

**Option A (Recommended):** From official repository

```bash
cd ~
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
```

This compiles CUDA operators against your current PyTorch 2.4.0 and CUDA toolchain.

**Option B:** PyPI wrapper (if you prefer pip-only)

```bash
pip install groundingdino-py==0.4.0
```

### Step 4: Install MobileSAM

MobileSAM is not available on PyPI and must be installed from GitHub:

```bash
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

### Step 5: Install Remaining Dependencies

```bash
cd ~/stcm-assistive-nav/stcm
pip install -r requirements.txt
```

### Step 6: Install ROS 2 Dependencies

```bash
# Make sure ROS 2 Humble is sourced
source /opt/ros/humble/setup.bash

# Install ROS dependencies
cd ~/stcm-assistive-nav
rosdep install --from-paths stcm --ignore-src -y
```

### Step 7: Build the ROS 2 Package

```bash
cd ~/stcm-assistive-nav
colcon build --packages-select stcm
```

### Step 8: Setup Your Shell Environment

For every new terminal session where you want to use this package:

```bash
# 1. Activate conda environment FIRST
conda activate stcm_env

# 2. Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# 3. Source your workspace
source ~/stcm-assistive-nav/install/setup.bash
```

**Tip:** Add this to your `~/.bashrc` for convenience:

```bash
# Add to ~/.bashrc
alias stcm_setup='conda activate stcm_env && source /opt/ros/humble/setup.bash && source ~/stcm-assistive-nav/install/setup.bash'
```

Then you can just run `stcm_setup` in any new terminal.

### Verify Installation

Quick sanity check:

```bash
python -c "import torch; from groundingdino.util.inference import load_model; print('GroundingDINO import OK')"
```

## Model checkpoints
The perception modules expect the pretrained weights to live under `~/.stcm/ckpts` (override with `STCM_CKPT_DIR`). The helper CLI downloads and renames everything for you:

```bash
ros2 run stcm stcm_download_checkpoints               # download all defaults into ~/.stcm/ckpts
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
- `target_labels` / `target_label_thresholds`: object classes of interest and per-class merge radius.
- `text_prompt`, `box_threshold`, `text_threshold`: detection prompt and thresholds.
- `graph_output_path` / `graph_input_path`: where semantic graphs are stored.

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
- Checkpoints: `~/.stcm/ckpts` (override via `STCM_CKPT_DIR`)
- RViz config: `stcm/config/semantic_mapping.rviz`
- Output graphs: configurable per node (`graph_output_path`)

## License
MIT License © 2025 Anhar Risnumawan. External models (GroundingDINO, MobileSAM, DepthAnything) retain their original licenses—review them before deployment.
