# CLAUDE.md

Index file for Claude Code (claude.ai/code) work in this repo.

> **Sync note:** `AGENTS.md` mirrors this file for Codex / other agent tools.
> Update both when editing — drift = confusion.

## Project Overview

**STCM Assistive Navigation** = ROS 2 Humble package implementing Semantic
Topological Cognitive Mapping. Combines GroundingDINO (open-vocab detect) +
MobileSAM (efficient seg) to build semantic graphs from RGB-D streams.
Graphs show spatial relations between objects, used for high-level robot
reason + navigation.

System processes synced RGB-D streams, detects objects via text prompts,
segs them, computes 3D world pose, maintains NetworkX graph tracking
objects + spatial relations over time.

## Package structure (high-level)

```
stcm/
├── stcm/
│   ├── core/          # Perception (GDINO, MobileSAM, Depth)
│   ├── nodes/         # ROS 2 nodes (builder, updater)
│   ├── tools/         # Checkpoint mgmt CLI
│   ├── image_listener.py  # RGB-D sync + TF
│   ├── map_utils.py       # Graph ops + JSON persistence
│   └── ros_utils.py
├── test/              # Standalone perception tests
├── config/            # Launch YAML
├── launch/            # ROS 2 launch files
└── imgs/              # Test images
```

Detail in `.claude/rules/perception-pipeline.md`.

## Where to find things

Operational detail lives in `.claude/rules/`. Load on demand:

- **Env setup, pip deps, checkpoints, system Python runtime**
  → `.claude/rules/env-setup.md`
- **Build (colcon), launch file usage, YAML params, ROS params**
  → `.claude/rules/build-and-launch.md`
- **Perception components, graph mgmt, builder/updater node internals,
  detection+merging logic, package-structure map**
  → `.claude/rules/perception-pipeline.md`
- **Topic inspection, RViz viz, threshold tuning, common failures**
  → `.claude/rules/debugging.md`
- **Paper eval protocol (scenarios, metrics, ablations, sensitivity)**
  → `.claude/rules/eval-protocol.md`
- **Runtime benchmark protocol (per-module latency, HW disclosure)**
  → `.claude/rules/benchmark-protocol.md`

Domain knowledge also in `.claude/skills/`:
- `env-bootstrap` — first-time fresh setup
- `ros2-debug` — topic/TF/sync diagnosis
- `stcm-tuning` — symptom → knob map (detection, GNG, place GNG)
- `eval-harness` — EDD framework for Table B/C/E numbers
- `benchmark` — runtime baseline + regression → Table D
- `python-testing` — pytest/parametrize for sensitivity sweep
- `pytorch-patterns` — reproducible GDINO/SAM benchmark
- `deep-research` — cited SOTA capability matrix → Table F
- `documentation-lookup` — verify Nav2/FAST-LIO2 API claims (Context7)
- `article-writing` — trim Intro/Related Work prose

Inner-loop commands in `.claude/commands/`:
- `/build`, `/clean-build` — colcon build
- `/launch` — `ros2 launch stcm semantic_mapping.launch.py`
- `/download-ckpts` — model weight downloader
- `/test-perception` — run standalone perception tests
- `/rosbag-replay` — offline sequential bag replay
- `/eval` — drive paper eval protocol (Tables B/C/E)
- `/verify` — submission preflight (figs dpi, acronyms, track-changes)
- `/quality-gate` — block until every AE-/R- item has evidence

Revision agents in `.claude/agents/` (invoke via Agent tool):
- `planner`, `python-reviewer`, `silent-failure-hunter`, `comment-analyzer`, `doc-updater`

## Paper revision workflow (JACIII Jc26-0002)

See `response_to_reviewers.pdf` for AE-1..AE-13 + R1/R2 items. Workflow:

1. Scaffold sensitivity + scenarios → `/eval --sweep` (AE-9, AE-1)
2. Per-module runtime → `benchmark` skill + `silent-failure-hunter` agent (AE-4)
3. Ablations + SOTA matrix → `/eval` paired runs + `deep-research` skill (AE-2, AE-3)
4. Figures ≥300 dpi ≥10pt, split Fig. 9 → `scripts/render_figures.py` (AE-6)
5. Trim Intro/Related Work + glossary → `article-writing` skill + `doc-updater` agent (AE-5, AE-7, AE-8, AE-10)
6. Submission preflight → `/verify` then `/quality-gate paper` (AE-11..13)

Harness rules (permissions, env, hooks) in `.claude/settings.json`.

## Quick start (assumes env ready)

```bash
# Build
colcon build --packages-select stcm
source install/setup.bash

# Launch
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=$(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping_params.yaml

# Override params
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=path/to/config.yaml \
  text_prompt:="chair . table ." \
  graph_output_path:=/tmp/my_graph.json
```

First-time setup: see `.claude/rules/env-setup.md` or skill `env-bootstrap`.

## Important invariants (do not break)

- ROS nodes must run on `/usr/bin/python3`. Keep shebangs.
- ML deps isolated under `PYTHONUSERBASE=$HOME/.local/stcm_sys_py310`.
  Export before source ROS.
- Text prompt classes each end `" ."` (space + period) for GroundingDINO.
- Each class in `target_labels` needs matching entry in
  `target_label_thresholds` (same length).
- Checkpoints default `./models` dir, override via `STCM_CKPT_DIR`.
- Builder node needs TF chain camera → base → world before processing.
- Graphs saved as JSON in NetworkX node-link format (readable + editable).
- Never commit `models/`, `output/`, `build/`, `install/`, `log/`. See
  `.gitignore`.
- When `use_sim_time` flag set, match to clock source (bag → true, live → false).

## Testing (quick ref)

```bash
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
source /opt/ros/humble/setup.bash
source install/setup.bash

python3 stcm/test/test_gdino_sam.py stcm/imgs/irvl-clutter-test.png
python3 stcm/test/test_depth_anything.py stcm/imgs/color-000089.png
```

Detail in `.claude/rules/debugging.md`.

## Harness overview

`.claude/settings.json` configures:
- **Permissions**: wildcard-allow daily ROS/colcon/test commands, deny writes
  to `models/`, `output/`, `build/`, `install/`, `log/`, `.env*`.
- **Env**: `PYTHONUSERBASE` exported for all tool calls.
- **Hooks**: PreToolUse blocks destructive `rm -rf` outside build dirs.
  PostToolUse runs `py_compile` on edited `.py`. Stop nudges verification.

Per-user overrides go in `.claude/settings.local.json` (gitignored).
