# STCM Planner: LLM From Semantic Graph

This guide shows how to drive the 2D RViz simulation from an STCM JSON graph.

## 1) Generate or locate an STCM graph

If you already have a graph, note its path and skip this step.

```bash
colcon build --packages-select stcm
source /opt/ros/humble/setup.bash
source install/setup.bash

# Run the mapper (update config file as needed)
ros2 launch stcm semantic_mapping.launch.py \
  config_file:=$(ros2 pkg prefix stcm)/share/stcm/config/semantic_mapping_params.yaml
```

The default graph output path is `output/stcm.json` (override with
`graph_output_path:=...` in the YAML or launch args).

## 2) Run the LLM simulator

```bash
colcon build --packages-select stcm_planner
source install/setup.bash

ros2 run stcm_planner semantic_graph_simulator --ros-args \
  -p graph_path:=/output/stcm.json \
  -p model:=gpt-4o \
  -p run_mode:=use_tools
```

Notes:
- `graph_path` points to your STCM JSON file.
- `model` and `run_mode` must match your configured LLM backend.
- The simulator listens for queries on `/stcm_planner_query`.

## 3) Send a query

```bash
ros2 run stcm_planner language_query_publisher
```

Type an instruction like:

```
Pick the bottle and put it into the trash bin.
```

## 4) RViz setup

Add these displays:
- MarkerArray: `/semantic_graph_sim/nodes`
- Marker: `/semantic_graph_sim/path`
- Marker: `/semantic_graph_sim/robot`

You should see the planned path and a moving robot marker as the LLM plan is
executed.
