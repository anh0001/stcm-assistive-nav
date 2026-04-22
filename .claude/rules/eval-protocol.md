# Eval Protocol Rules (JACIII Jc26-0002 Revision)

Reproducibility contract for Tables A/B/C/E numbers. Authoritative for `/eval`.

## Scenarios (AE-1 / R1-1)

Three independent indoor sessions. Each ≥3 runs w/ fresh seed.

| ID | Name | Bag path |
|----|------|----------|
| S1 | corridor (TMU 7F Bldg 2) | `data/bags/corridor_*.db3` |
| S2 | office/lab room | `data/bags/office_*.db3` |
| S3 | meeting room | `data/bags/meeting_*.db3` |

Dataset card → Table A: duration, path length, label vocab, ground-truth instance count.

## Metrics

**Navigation** (per goal trial):
- success rate ± Wilson 95% CI
- completion time (s)
- path-length ratio (actual / shortest graph path)
- ≥10 trials per scenario, goals drawn from {table, chair, trash bins, water fountain, door, desk}

**Semantic labeling** (per stable node):
- macro-F1, micro-F1 on committed labels vs GT vocab
- LabelCov = |nodes with max score > τ_text| / |nodes|
- position stability σ (m), threshold 0.15 m

**Grounding** (per command):
- exact-match + top-2 accuracy
- command sets: simple (label-only), disambiguation (multi-instance), compositional (spatial-relational)
- ≈15 commands/scenario → ≈45 total

## Ablations (AE-2, AE-3)

Paired runs on identical bag+commands:
- `full` — dual-GNG + LLM
- `semantic-only` — no topology
- `place-gng-only` — no instance-GNG
- `no-llm` — template string match

Report McNemar p-value per command subset. Effect size = Δ accuracy w/ 95% CI.

## Sensitivity sweep (AE-9)

| Param | Values |
|-------|--------|
| D_new | 0.3, 0.5, 0.8 |
| α | 0.05, 0.1, 0.2 |
| τ_box = τ_text | 0.45, 0.55, 0.65 |
| η_min | 1, 3, 5 |

Hold others at Table 1 defaults. One metric per cell (macro-F1). Output → Table E.

## Reproducibility

Every result JSON records:
- git SHA of stcm commit
- rosbag path + sha256
- checkpoint sha256 (GDINO, MobileSAM)
- numpy/torch seed, CUDA device
- wall-clock timestamp + hostname
- launch config YAML snapshot

Fail-loud rules:
- Never mock ROS topics for paper numbers.
- Never merge runs across git SHAs.
- If FAST-LIO2 TF gap > 0.2s, discard run.
- If GDINO returns 0 detections for >10% frames, flag run + keep for ablation note.

## Output layout

```
results/
  eval/
    <scenario>_<run>_<sha>.json
    llm_ablation_<sha>.json
    sensitivity_<sha>.json
  bench/
    runtime_<sha>.json
paper/
  tables/{A,B,C,D,E,F,G}.tex    # generated from results/*
```

LaTeX tables must be generated, not hand-typed. `/verify` blocks on hand edits.
