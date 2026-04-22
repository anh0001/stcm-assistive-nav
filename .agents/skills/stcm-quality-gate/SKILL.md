---
name: stcm-quality-gate
description: Check whether STCM reviewer experiment evidence is complete, reproducible, and free of silent-failure flags.
origin: project
---

# STCM Quality Gate

Use this skill before claiming that reviewer experiment evidence is ready.

## Command

```bash
python3 scripts/experiments/quality_gate.py
```

## Gate Meaning

- Dataset summary exists for every scenario and required topics are present.
- Full STCM runs exist for every scenario.
- Ablation variants exist for every scenario: `full`, `semantic-only`,
  `place-gng-only`, and `no-llm`.
- Runtime timings include at least GroundingDINO, SAM, pose association, and
  graph update measurements.
- Sensitivity evidence exists for every manifest sweep key.

If the gate blocks, run only the concrete commands it prints. Do not mark a
reviewer item complete on memory or on files in `output/`.

