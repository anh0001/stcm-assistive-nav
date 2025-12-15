# STCM Assistive Navigation — Agent Handbook

Use this document whenever you (or another AI coding assistant) contribute to the workspace. It complements `README.md`, `CLAUDE.md`, and `.github/copilot-instructions.md` by summarizing expectations that apply to *every* agent.

## 1. Mission Snapshot
- ROS 2 Humble package that fuses GroundingDINO + MobileSAM (optionally Depth Anything) to build/persist semantic graphs from RGB‑D streams.
- Runtime nodes live under `stcm/stcm/nodes/`; perception primitives are inside `stcm/stcm/core/`.
- Launch entry point: `stcm/launch/semantic_mapping.launch.py`, usually fed by `stcm/config/semantic_mapping_params.yaml`.
- Graphs store labeled objects, 3‑D centroids, and relationship metadata; they publish RViz `MarkerArray` data on `/semantic_graph/nodes`.

## 2. Environment & Runtime Rules
1. **Always** activate Conda before ROS: `conda activate stcm_env` → `source /opt/ros/humble/setup.bash` → `source ./install/setup.bash`.
2. Build with `colcon build --packages-select stcm`; clean via `rm -rf build install log`.
3. Model checkpoints default to `./models` (override with `$STCM_CKPT_DIR`). Run `ros2 run stcm stcm_download_checkpoints` to hydrate weights.
4. When launching, prefer YAML configs, but expose new parameters in both the config file and launch description.

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

Welcome aboard—ship changes that keep the perception stack reproducible, configurable, and well-tested.***
