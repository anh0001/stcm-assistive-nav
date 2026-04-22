# Codex Agents Assets

This directory is the Codex/plugin-style skill surface for this repo.

The source `.claude/` assets remain available for Claude Code, but Codex-facing
reusable workflows live here under `.agents/skills/*/SKILL.md` with optional
`agents/openai.yaml` metadata.

Current project-local skills:

- `stcm-experiment-harness` — run reviewer experiment matrices and collect evidence.
- `stcm-quality-gate` — check whether experiment evidence is complete.
- `stcm-ros2-debug` — diagnose ROS 2 topic/TF/sync failures.
- `stcm-tuning` — map graph/perception symptoms to tuning knobs.

