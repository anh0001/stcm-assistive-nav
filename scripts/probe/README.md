# GroundingDINO Label Recognition Probe

Diagnose which outdoor S3 GT classes GroundingDINO actually fires on, and find
better text aliases for the ones it misses.

## Why

DSE+audit (2026-05-07) showed F1 plateau at 0.30 with these GT classes
persistently FN: `electric_wheelchair`, `basketball`, `umbrella`,
`electric_scooter`, `prohibition_sign`, `pair_of_ramps`, 5/6 trailers
(southwest poses).

Two possible causes:
1. Robot path never gets close enough to those objects → recall ceiling is
   trajectory-bound, not perception.
2. GDINO doesn't recognize the prompt text → fix is alias rewording.

This probe distinguishes (1) vs (2):
- `dump_frames.py` extracts images where the robot WAS within `--max-dist-m`
  of any GT pose, so we know each missed class should at least be visible.
- `gdino_probe.py` runs GDINO on those images with both production prompt
  bank and an "extras" alias YAML, logs hits + scores per (image, alias).

## Workflow

```bash
# Source env first (always)
export PYTHONUSERBASE="$HOME/.local/stcm_sys_py310"
source /opt/ros/humble/setup.bash
source install/setup.bash

# Step A — dump frames near every GT object (filter <=6m from any GT)
python3 scripts/probe/dump_frames.py \
  --bag /mnt/STREAM/outdoor_livinglab_01_20260501_162329_0 \
  --image-topic /camera/image_raw \
  --output-dir scripts/probe/frames/outdoor \
  --gt-json configs/experiments/ground_truth/outdoor_livinglab_stcm_gt.json \
  --max-dist-m 6.0 \
  --every-n 3 --max-frames 80

# Step B — probe with production bank + alias extras
python3 scripts/probe/gdino_probe.py \
  --image-dir scripts/probe/frames/outdoor \
  --prompt-bank stcm/config/outdoor_livinglab_nyu_grounded_prompts.yaml \
  --extra-aliases scripts/probe/outdoor_extra_aliases.yaml \
  --output-csv scripts/probe/results/probe_outdoor.csv \
  --max-images 80
```

## Output: per-label hit-rate analysis

```python
import pandas as pd
df = pd.read_csv("scripts/probe/results/probe_outdoor.csv")
# Hit rate per class+alias
df.groupby(["class","alias"])["hit"].agg(["sum","count","mean"]).sort_values("mean", ascending=False)
```

Best alias = highest `mean` hit rate. Aliases with `mean >= 0.5` across the
N filtered frames are strong candidates to swap into
`stcm/config/outdoor_livinglab_nyu_grounded_prompts.yaml`.

## Production fidelity

`gdino_probe.py` replicates `nyu_grounded_backend.py:detect_and_segment`:
- Same model: `IDEA-Research/grounding-dino-base` via
  `transformers.AutoModelForZeroShotObjectDetection`.
- Same prompt format: lowercased period-joined aliases (see `_build_prompt`).
- Same per-class threshold pair grouping (chunks).
- Same post-processing (`post_process_grounded_object_detection`).
- **Skips CLIP reranking** — that's a separate dimension and can silently
  discard valid GDINO hits. To diagnose pure GDINO recognition, this is
  intentional.

## Files

| File | Purpose |
|------|---------|
| `dump_frames.py` | Extract sampled frames + robot pose sidecars from rosbag |
| `gdino_probe.py` | Run GDINO on frames; sweep production bank + extra aliases |
| `outdoor_extra_aliases.yaml` | Extra alias variants to try per failing class |
| `frames/outdoor/` | Output: PNG + JSON sidecars (filled by `dump_frames.py`) |
| `results/probe_outdoor.csv` | Output: per-(image, alias) hit/score table |
