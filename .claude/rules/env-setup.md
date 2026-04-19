# Environment Setup Rules

Authoritative env detail. `CLAUDE.md` points here.

## Python runtime

- ROS nodes run on `/usr/bin/python3`. Keep `#!/usr/bin/python3` shebang.
- Skip conda / venv. ML deps live under dedicated `PYTHONUSERBASE`.

```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
python3 -m pip install --upgrade --user pip
python3 -m pip install --user ros2-numpy
python3 -m pip install --user torch torchvision torchaudio
```

Follow PyTorch selector pick right CUDA wheels. Driver version > toolkit
version matters — binaries ship CUDA runtime.

Export `PYTHONUSERBASE` BEFORE source ROS so deps resolve:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

## System Python runtime checklist

- Keep ROS + perception on `/usr/bin/python3` w/ `PYTHONUSERBASE=$HOME/.local/stcm_sys_py310`.
- Export `PYTHONUSERBASE` before source ROS or VS Code tasks/launchers. Bake
  into shell profile or wrapper scripts.
- Install new dep with `python3 -m pip install --user ...` while user base
  active. Never add pkgs to global `~/.local`.
- If split perception into helper service, start from same interpreter.
  Doc IPC params (`perception_endpoint`, `ipc_transport`, `request_timeout`)
  in YAML + launch files.
- Pros: ROS-friendly, min ABI fights. Cons: ML deps maintained in
  system-python land — stay isolated inside `PYTHONUSERBASE`.

## Dependencies

**Python (pip into `$PYTHONUSERBASE`):**
- PyTorch 2.4.0 + CUDA 12.1
- GroundingDINO (source via `pip install -e` or `groundingdino-py`)
- MobileSAM (from GitHub)
- HuggingFace `transformers`, `huggingface-hub`
- Vision: `opencv-python`, `Pillow`, `supervision`, `open3d`, `scikit-image`
- Graph: `networkx`
- ROS bridge: `ros2-numpy`

**ROS 2 (rosdep):**
- `rclpy`, `sensor_msgs`, `geometry_msgs`, `visualization_msgs`
- `tf2_ros`, `tf_transformations`
- `cv_bridge`, `message_filters`

**System:** Ubuntu 22.04, ROS 2 Humble, CUDA 12.x + cuDNN 9.

## Model checkpoints

Default `./models` (override `STCM_CKPT_DIR`):

```bash
ros2 run stcm stcm_download_checkpoints
ros2 run stcm stcm_download_checkpoints --list
ros2 run stcm stcm_download_checkpoints --models mobilesam --target /data/ckpts
export STCM_CKPT_DIR=/data/ckpts
```

Expected layout:
- `models/gdino/groundingdino_swint_ogc.pth`
- `models/mobilesam/vit_t.pth`
- `models/depth_anything/depth_anything_vitb14.pth` (optional)
