# Experiment Audit Trace

**Date**: 2026-05-07
**Run**: 01
**Reviewer**: gpt-5.5 (Codex MCP, model_reasoning_effort=xhigh, sandbox=read-only)
**Caller**: Claude Code (Opus 4.7 1M)
**Audit target**: F1=0.300 from DSE iter 18 on `outdoor_livinglab × full-nyu-proposals`
**Codex thread**: 019dffa3-a568-7723-92b5-9c309e02b23c

## Verdict
- Overall: **FAIL**
- Inflation: 1.20x (strict 0.250 vs alias 0.300)
- 5 fail/warn checks: A (GT provenance), D (dead knobs), E (scope), F (label aliasing), G (range gate NaN bug). B warn. C pass.

## Outputs
- /home/anhar/codes/stcm-assistive-nav/EXPERIMENT_AUDIT.md
- /home/anhar/codes/stcm-assistive-nav/EXPERIMENT_AUDIT.json
