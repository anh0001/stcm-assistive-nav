---
name: env-bootstrap
description: First-time setup of STCM environment (PYTHONUSERBASE, pip deps, ROS source, checkpoint download). Fire when user says "fresh setup", "new machine", "first time", "install failed", "ModuleNotFoundError", "cannot find torch", "checkpoint missing", "PYTHONUSERBASE", or when they ask how to get the project running end-to-end from scratch.
---

# STCM Environment Bootstrap

Isolate ML deps under dedicated user base so system Python stays ROS-friendly.

## 1. Export PYTHONUSERBASE (persistent)

Add to `~/.bashrc` (or `~/.zshrc`):

```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
export PATH="$PYTHONUSERBASE/bin:$PATH"
```

Re-source: `source ~/.bashrc`. Confirm: `echo $PYTHONUSERBASE`.

Why dedicated base: keep ROS `/usr/bin/python3` clean, isolate PyTorch/CUDA
wheels, avoid clashing w/ other projects in `~/.local`.

## 2. Install pip deps into user base

```bash
python3 -m pip install --upgrade --user pip
python3 -m pip install --user ros2-numpy
python3 -m pip install --user torch torchvision torchaudio  # pick CUDA wheels via PyTorch selector
python3 -m pip install --user \
  opencv-python pillow supervision open3d scikit-image \
  networkx transformers huggingface-hub
# GroundingDINO: clone + pip install -e, or use groundingdino-py
# MobileSAM: pip install from GitHub
```

Check wheel CUDA matches NVIDIA driver (driver ≥ CUDA runtime from wheel).

## 3. Source ROS + workspace (every shell)

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash   # after first colcon build
```

`PYTHONUSERBASE` must be exported BEFORE sourcing ROS, else pip deps invisible.

## 4. Download checkpoints

```bash
ros2 run stcm stcm_download_checkpoints
# or override target: --target /data/ckpts + export STCM_CKPT_DIR=/data/ckpts
```

Expected layout:
- `models/gdino/groundingdino_swint_ogc.pth`
- `models/mobilesam/vit_t.pth`
- `models/depth_anything/depth_anything_vitb14.pth` (optional)

## 5. Build package

```bash
colcon build --packages-select stcm
source install/setup.bash
```

## 6. Smoke test

```bash
python3 stcm/test/test_gdino_sam.py stcm/imgs/irvl-clutter-test.png
```

Output = annotated PNG. If fails here → models or pip deps broken, not ROS.

## Gotchas

1. **Shebang must stay `/usr/bin/python3`** in ROS nodes — switching to
   conda/venv interpreter breaks ROS's Python discovery.
2. **Never install pkgs to global `~/.local`** w/o `PYTHONUSERBASE` set —
   pollutes base user-site, hard to isolate later.
3. **CUDA 12.1 wheels need driver ≥ 525** — check `nvidia-smi` first.
4. **Rerun `source install/setup.bash` after every `colcon build`** — stale
   env misses new entry points.
5. **Run tests from repo root** — they use relative paths to `stcm/imgs/`.
6. **If using VS Code tasks:** bake `PYTHONUSERBASE` into the task env, not
   just your shell profile, else tasks run w/o deps.

## Verification checklist

- [ ] `echo $PYTHONUSERBASE` = `~/.local/stcm_sys_py310`
- [ ] `python3 -c 'import torch; print(torch.cuda.is_available())'` → `True`
- [ ] `ros2 pkg list | grep stcm` returns `stcm`
- [ ] `ls models/gdino/groundingdino_swint_ogc.pth` exists
- [ ] `test_gdino_sam.py` produces annotated output
