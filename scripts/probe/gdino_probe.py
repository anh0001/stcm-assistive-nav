#!/usr/bin/env python3
"""Probe GroundingDINO with the production NYU prompt bank + alternative aliases.

Replicates the EXACT call used by `nyu_grounded_backend.py:detect_and_segment`:
- Same model: `IDEA-Research/grounding-dino-base` via `transformers.AutoModelForZeroShotObjectDetection`
- Same prompt format: lowercased period-joined aliases (`build_prompt`)
- Same per-class threshold pair grouping
- Same post-processing (`post_process_grounded_object_detection`)

Diagnoses: which GT classes does GDINO actually fire on, and which aliases
yield the best score. Skips CLIP reranking — that's a separate dimension.

Usage (production aliases):
    python3 scripts/probe/gdino_probe.py \
        --image-dir scripts/probe/frames/outdoor \
        --prompt-bank stcm/config/outdoor_livinglab_nyu_grounded_prompts.yaml \
        --output-csv scripts/probe/results/probe_outdoor_baseline.csv

Usage (probe extra aliases on failing classes):
    python3 scripts/probe/gdino_probe.py \
        --image-dir scripts/probe/frames/outdoor \
        --prompt-bank stcm/config/outdoor_livinglab_nyu_grounded_prompts.yaml \
        --extra-aliases scripts/probe/outdoor_extra_aliases.yaml \
        --output-csv scripts/probe/results/probe_outdoor_with_extras.csv

`--extra-aliases` YAML schema (one entry per CLASS LABEL to probe; runs each
listed alias as its own SOLO prompt at given thresholds):
    classes:
      basketball:
        thresholds: [0.18, 0.18]
        aliases:
          - "orange basketball"
          - "round orange ball on the ground"
          - "sports ball"
          - "ball"
      umbrella:
        thresholds: [0.20, 0.20]
        aliases:
          - "outdoor umbrella"
          - "patio parasol"

Output CSV columns:
    image, class, alias, threshold_box, threshold_text, score, x1, y1, x2, y2, hit

`hit=1` if GDINO returned at least one box for this prompt above threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]


def _load_prompt_bank(path: Path) -> list[dict]:
    """Return list of (class_label, aliases, box_thresh, text_thresh) groups.

    Mirrors how `nyu_grounded_backend._build_detection_chunks` groups classes
    sharing the same threshold pair within a chunk into a single GDINO call.
    """
    bank = yaml.safe_load(path.read_text()) or {}
    chunks = bank.get("chunks", {}) or {}
    detection_groups: list[dict] = []
    for chunk_name, chunk in chunks.items():
        default_box, default_text = (chunk.get("thresholds", [0.30, 0.30]) + [0.30, 0.30])[:2]
        # Group by effective threshold pair within chunk
        groups: dict[tuple[float, float], list[tuple[str, list[str]]]] = {}
        for cls in chunk.get("classes", []) or []:
            label = cls["label"]
            aliases = cls.get("detect_aliases") or cls.get("aliases") or [label]
            tb = float(cls.get("box_threshold", default_box))
            tt = float(cls.get("text_threshold", default_text))
            groups.setdefault((tb, tt), []).append((label, aliases))
        for (tb, tt), members in groups.items():
            detection_groups.append({
                "chunk": chunk_name,
                "box_threshold": tb,
                "text_threshold": tt,
                "members": members,
            })
    return detection_groups


def _build_prompt(aliases: list[str]) -> str:
    """Replicate `build_prompt`: lowercase, period-joined, no trailing space-period."""
    parts = [a.strip().lower().rstrip(".") for a in aliases if a.strip()]
    return ". ".join(parts) + "."


def _load_extra_aliases(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("classes", {}) or {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-dir", required=True, type=Path)
    ap.add_argument("--prompt-bank", required=True, type=Path)
    ap.add_argument("--extra-aliases", type=Path, default=None)
    ap.add_argument("--output-csv", required=True, type=Path)
    ap.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    ap.add_argument("--device", default="cuda",
                    help="cuda or cpu; will fall back to cpu if cuda unavailable")
    ap.add_argument("--max-images", type=int, default=0,
                    help="Limit number of images probed (0 = all)")
    ap.add_argument("--mode", choices=["production", "extras", "both"],
                    default="both",
                    help="production = bank groups as-is; extras = each extra alias as solo prompt")
    args = ap.parse_args()

    image_paths = sorted(args.image_dir.glob("frame_*.png"))
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise SystemExit(f"No frame_*.png in {args.image_dir}")

    # Lazy heavy imports
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[probe] device={device}, images={len(image_paths)}, model={args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device).eval()

    bank_groups = _load_prompt_bank(args.prompt_bank)
    extra = _load_extra_aliases(args.extra_aliases)
    print(f"[probe] prompt bank: {len(bank_groups)} groups; extras: {len(extra)} classes")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["image", "mode", "class", "alias", "prompt_text",
              "threshold_box", "threshold_text",
              "n_detections", "max_score", "first_box_x1y1x2y2", "hit"]
    with args.output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for img_idx, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert("RGB")
            print(f"  [{img_idx+1}/{len(image_paths)}] {img_path.name}")

            if args.mode in ("production", "both"):
                for grp in bank_groups:
                    aliases = []
                    label_for_grp = "+".join(m[0] for m in grp["members"])
                    for _, alias_list in grp["members"]:
                        aliases.extend(alias_list)
                    prompt = _build_prompt(aliases)
                    inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = gdino(**inputs)
                    results = processor.post_process_grounded_object_detection(
                        out, inputs.input_ids,
                        threshold=grp["box_threshold"],
                        text_threshold=grp["text_threshold"],
                        target_sizes=[(img.height, img.width)],
                    )[0]
                    scores = results.get("scores")
                    boxes = results.get("boxes")
                    n = int(scores.shape[0]) if scores is not None else 0
                    max_score = float(scores.max().item()) if n > 0 else 0.0
                    first_box = (boxes[0].tolist() if n > 0 else None)
                    w.writerow({
                        "image": img_path.name,
                        "mode": "production",
                        "class": label_for_grp,
                        "alias": "ALL_GROUP",
                        "prompt_text": prompt,
                        "threshold_box": grp["box_threshold"],
                        "threshold_text": grp["text_threshold"],
                        "n_detections": n,
                        "max_score": round(max_score, 4),
                        "first_box_x1y1x2y2": json.dumps(first_box) if first_box else "",
                        "hit": 1 if n > 0 else 0,
                    })

            if args.mode in ("extras", "both"):
                for cls_label, spec in extra.items():
                    tb, tt = (spec.get("thresholds", [0.20, 0.20]) + [0.20, 0.20])[:2]
                    for alias in spec.get("aliases", []):
                        prompt = _build_prompt([alias])
                        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            out = gdino(**inputs)
                        results = processor.post_process_grounded_object_detection(
                            out, inputs.input_ids,
                            threshold=tb, text_threshold=tt,
                            target_sizes=[(img.height, img.width)],
                        )[0]
                        scores = results.get("scores")
                        boxes = results.get("boxes")
                        n = int(scores.shape[0]) if scores is not None else 0
                        max_score = float(scores.max().item()) if n > 0 else 0.0
                        first_box = (boxes[0].tolist() if n > 0 else None)
                        w.writerow({
                            "image": img_path.name,
                            "mode": "extras",
                            "class": cls_label,
                            "alias": alias,
                            "prompt_text": prompt,
                            "threshold_box": tb,
                            "threshold_text": tt,
                            "n_detections": n,
                            "max_score": round(max_score, 4),
                            "first_box_x1y1x2y2": json.dumps(first_box) if first_box else "",
                            "hit": 1 if n > 0 else 0,
                        })

    print(f"[probe] wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
