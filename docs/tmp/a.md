# STCM Assistive Navigation — Agent Handbook

Use this document whenever you (or another AI coding assistant) contribute to the workspace. It complements `README.md`, `CLAUDE.md`, and `.github/copilot-instructions.md` by summarizing expectations that apply to *every* agent.

## 1. Mission Snapshot
- ROS 2 Humble package that fuses GroundingDINO + MobileSAM (optionally Depth Anything) to build/persist semantic graphs from RGB‑D streams.
- Runtime nodes live under `stcm/stcm/nodes/`; perception primitives are inside `stcm/stcm/core/`.
- Launch entry point: `stcm/launch/semantic_mapping.launch.py`, usually fed by `stcm/config/semantic_mapping_params.yaml`.
- Graphs store labeled objects, 3‑D centroids, and relationship metadata; they publish RViz `MarkerArray` data on `/semantic_graph/nodes`.
- The optional topological place graph (paper GNG) learns place nodes from the robot pose stream, publishes `MarkerArray` data on `/semantic_graph/place_graph`, and saves JSON to `place_gng_output_path`. Nodes appear only when the robot moves beyond `place_gng_distance_threshold` (default 1.5 m).

## 2. Environment & Runtime Rules
1. Keep ROS nodes on `/usr/bin/python3` (existing shebang) so rclpy uses the distro interpreter—skip conda entirely.
2. Isolate ML dependencies with a dedicated user base instead of polluting a “dirty” `~/.local`:
   ```bash
   export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
   python3 -m pip install --upgrade --user pip
   python3 -m pip install --user ros2-numpy
   python3 -m pip install --user torch torchvision torchaudio
   ```
   Extend this list as new imports appear; follow PyTorch’s selector for CUDA wheels (they ship their own CUDA runtime, so the driver version is what matters).
3. Ensure `PYTHONUSERBASE` is exported in every terminal (profile, launch wrapper, VS Code task, etc.) **before** sourcing ROS: `source /opt/ros/humble/setup.bash && source ./install/setup.bash`.
4. Build with `colcon build --packages-select stcm`; clean via `rm -rf build install log`. Prefer YAML configs for launches and expose new parameters in both the config and launch file.

## 3. Coding Expectations
- Python 3.10, ROS 2 `rclpy` nodes, MIT license.
- Match existing parameter names (snake_case), log via `self.get_logger()`, and respect the perception pipeline separation (listener → predictors → graph utils).
- Keep detection thresholds/toggles configurable; avoid hard‑coding robot-specific topics or frames.
- Add concise comments only when behavior is non-obvious; follow repository style (no type-ignores unless justified).
- Tests live in `stcm/test/`; expand them if you touch predictors, checkpoint logic, or graph utilities.

## 4. Common Tasks Cheat Sheet
| Task | Commands |
| --- | --- |
| Build + source | `colcon build --packages-select stcm && source install/setup.bash` |
| Launch full stack | `ros2 launch stcm semantic_mapping.launch.py config_file:=...` |
| Run builder manually | `ros2 run stcm semantic_map_builder --ros-args -p text_prompt:="chair . table ." -p graph_output_path:=/tmp/map.json` |
| Run updater | `ros2 run stcm semantic_map_updater --ros-args -p graph_input_path:=/tmp/map.json` |
| Download checkpoints | `ros2 run stcm stcm_download_checkpoints --list` / `--models mobilesam` |
| Standalone perception test | `python stcm/test/test_gdino_sam.py stcm/imgs/irvl-clutter-test.png` |

## 5. Debug & Integration Tips
- Ensure synchronized RGB, depth, and CameraInfo topics; `ImageListener` blocks until intrinsics arrive.
- TF frames must exist for `camera_frame → base_frame → world_frame`. Missing transforms halt pose projection.
- Graph churn usually means thresholds are too low or `filter_large_boxes` removes crucial detections; tune in YAML before touching code.
- If ROS env is unsourced, `message_filters` throws sync errors and parameters will be unset—double-check shell setup first.

## 6. Contribution Workflow
1. Sync checkpoints + environment.
2. Implement change with small, reviewable commits; keep modifications within relevant submodules.
3. Rebuild, rerun targeted tests/launch files.
4. Document new config keys or behaviors in `README.md`/`CLAUDE.md`/`copilot-instructions.md` if they affect operations.

## 7. When in Doubt
- Consult `README.md` for installation nuances, `CLAUDE.md` for deep architecture notes, and `.github/copilot-instructions.md` for ROS parameter expectations.
- Ask for clarification (or leave TODOs) when robot-specific assumptions leak into general-purpose code.

## 8. System Python Runtime
- Keep ROS + perception code on the distro interpreter with the `PYTHONUSERBASE` above. Leave the `#!/usr/bin/python3` shebang untouched.
- Install every imported package into that user base only; do **not** add more packages to the global `~/.local`.
- Export `PYTHONUSERBASE` in your shell profile, Launch task, or wrapper script so ROS nodes always see the isolated site-packages.
- Heavy perception helpers can still run out-of-process, but they should use the same Python install to avoid ABI mismatch.
- Pros: ROS-friendly and low-weirdness. Cons: you own the ML deps in system-python land, so keep the user base tidy.

Welcome aboard—ship changes that keep the perception stack reproducible, configurable, and well-tested.***
