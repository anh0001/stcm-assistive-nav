#!/usr/bin/env bash
# Run the full reviewer-driven eval: 2 scenes × 3 processing_period sweeps,
# then score F1/IoU/nav/grounding, and render LaTeX tables.
#
# Prereqs: env sourced (PYTHONUSERBASE, /opt/ros/humble, install/setup.bash),
#          rosbags accessible at paths listed in configs/experiments/manifest.yaml,
#          GT graphs in configs/experiments/ground_truth/.
#
# Usage:  bash scripts/eval/run_full_eval.sh [--no-run] [--scenes meeting livinglab]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SCENES=("meeting" "livinglab")
# Sweep over box_threshold rather than processing_period: STCM is deterministic
# in graph building once frames are fixed, so processing_period gives sigma=0.
# box_threshold genuinely perturbs detection counts -> real run-to-run variation.
BOX_THRESHOLDS=("0.30" "0.35" "0.40")
VARIANT="${VARIANT:-full-nyu-proposals}"
GOAL_LABELS_meeting=(chair "meeting table set" door "trash bin" "water fountain")
GOAL_LABELS_livinglab=(chair "cardboard box" "trash bin" "vacuum cleaner" door)

NO_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-run) NO_RUN=1; shift ;;
    --scenes) shift; SCENES=(); while [[ $# -gt 0 && "$1" != --* ]]; do SCENES+=("$1"); shift; done ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p paper/tables results/eval

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"

for scene in "${SCENES[@]}"; do
  GT="configs/experiments/ground_truth/${scene}_stcm_gt.json"
  if [[ ! -f "$GT" ]]; then
    echo "missing GT: $GT" >&2; exit 1
  fi

  CMDS="configs/eval/commands_${scene}.yaml"
  if [[ ! -f "$CMDS" ]]; then
    echo "missing commands yaml: $CMDS" >&2; exit 1
  fi

  goals_var="GOAL_LABELS_${scene}[@]"
  GOALS=("${!goals_var}")

  for i in "${!BOX_THRESHOLDS[@]}"; do
    bt="${BOX_THRESHOLDS[$i]}"
    run_id="${scene}_run${i}_bt${bt}_${GIT_SHA}"
    run_dir="results/eval/${run_id}"
    mkdir -p "$run_dir"

    echo "=== ${run_id} ==="

    if [[ "$NO_RUN" -eq 1 ]]; then
      echo "  --no-run: skipping rosbag replay"
    else
      python3 scripts/experiments/run_experiment.py \
        --scenario "$scene" \
        --variant "$VARIANT" \
        --sensitivity "box_threshold=${bt}" \
        --skip-bag-hash \
        --timeout-sec 1800 \
        || { echo "run_experiment failed for $run_id"; continue; }

      # locate latest stcm.json artifact for this scenario (timestamp-prefixed dirs)
      latest_artifact="$(ls -td results/eval/artifacts/**${scene}_${VARIANT}*/ 2>/dev/null | head -n1)"
      if [[ -z "${latest_artifact:-}" || ! -f "${latest_artifact}stcm.json" ]]; then
        echo "no stcm.json artifact found; skip scoring for $run_id" >&2
        continue
      fi
      cp "${latest_artifact}stcm.json" "${run_dir}/stcm.json" || { echo "cp failed for $run_id" >&2; continue; }
    fi

    pred="${run_dir}/stcm.json"
    if [[ ! -f "$pred" ]]; then
      echo "no prediction at $pred; skip scoring" >&2
      continue
    fi

    python3 scripts/experiments/benchmark_stcm_graph.py \
      --prediction "$pred" \
      --ground-truth "$GT" \
      --output-json "${run_dir}/benchmark.json" \
      --output-csv "${run_dir}/benchmark.csv" \
      --match-threshold-m 1.0 \
      --label-aliases configs/eval/label_aliases.json || true

    python3 scripts/eval/per_label_metrics.py \
      --benchmark "${run_dir}/benchmark.json" \
      --output "${run_dir}/per_label.json" || true

    python3 scripts/eval/nav_sim.py \
      --prediction "$pred" \
      --ground-truth "$GT" \
      --output "${run_dir}/nav.json" \
      --goal-labels "${GOALS[@]}" \
      --n-trials 15 \
      --seed "$i" \
      --label-radius 2.0 \
      --terminal-radius 2.5 || true

    python3 scripts/eval/grounding.py \
      --prediction "$pred" \
      --ground-truth "$GT" \
      --commands "$CMDS" \
      --output "${run_dir}/grounding.json" \
      --match-radius 1.5 || true
  done

  # stability across the 3 runs for this scene
  python3 scripts/eval/stability.py \
    --predictions results/eval/${scene}_run*_${GIT_SHA}/stcm.json \
    --output "results/eval/${scene}_stability_${GIT_SHA}.json" || true
done

python3 scripts/eval/dataset_card.py \
  --output paper/tables/dataset_card.json \
  --scenes "${SCENES[@]}"

python3 scripts/eval/render_tables.py \
  --eval-root results/eval \
  --scenes "${SCENES[@]}" \
  --dataset-card paper/tables/dataset_card.json \
  --output-dir paper/tables

echo "done. tables in paper/tables/{A,B,C,E}.tex"
