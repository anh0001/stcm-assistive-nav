"""Helpers for proposal label re-ranking and geometry-aware label priors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LabelDecision:
    label: str | None
    confidence: float
    margin: float
    label_scores: dict[str, float]


def normalize_score(value: float | int | None) -> float:
    if value is None:
        return 0.0
    score = float(value)
    if score < 0.0:
        score = (score + 1.0) * 0.5
    return min(max(score, 0.0), 1.0)


def combine_label_scores(
    *,
    detector_label: str,
    detector_score: float | int | None,
    crop_scores: dict[str, float] | None,
    image_priors: dict[str, float] | None,
) -> dict[str, float]:
    labels = set()
    if crop_scores:
        labels.update(crop_scores)
    if image_priors:
        labels.update(image_priors)
    labels.add(str(detector_label))

    detector_score_norm = normalize_score(detector_score)
    combined: dict[str, float] = {}
    for label in labels:
        total = normalize_score((crop_scores or {}).get(label))
        total += normalize_score((image_priors or {}).get(label))
        if label == detector_label:
            total += detector_score_norm
        combined[label] = total / 3.0
    return combined


def choose_label(label_scores: dict[str, float] | None, min_margin: float) -> LabelDecision:
    if not label_scores:
        return LabelDecision(label=None, confidence=0.0, margin=0.0, label_scores={})

    ordered = sorted(
        ((str(label), normalize_score(score)) for label, score in label_scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    top_label, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = top_score - second_score
    if top_score <= 0.0 or margin < float(min_margin):
        return LabelDecision(
            label=None,
            confidence=top_score,
            margin=margin,
            label_scores={label: score for label, score in ordered},
        )
    return LabelDecision(
        label=top_label,
        confidence=top_score,
        margin=margin,
        label_scores={label: score for label, score in ordered},
    )


def mask_area_fraction(mask: np.ndarray | None) -> float:
    if mask is None:
        return 0.0
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
        mask_arr = mask_arr[0]
    if mask_arr.size == 0:
        return 0.0
    return float(mask_arr.mean())


def apply_geometry_priors(
    *,
    label_scores: dict[str, float] | None,
    pose: np.ndarray | list[float] | tuple[float, ...] | None,
    mask: np.ndarray | None,
    priors: dict[str, dict[str, Any]] | None,
) -> dict[str, float]:
    if not label_scores or not priors:
        return dict(label_scores or {})

    pose_arr = None
    if pose is not None:
        pose_arr = np.asarray(pose, dtype=float).reshape(-1)
    z_value = float(pose_arr[2]) if pose_arr is not None and pose_arr.size >= 3 else None
    area_frac = mask_area_fraction(mask)

    adjusted = {label: normalize_score(score) for label, score in label_scores.items()}
    for label, score in list(adjusted.items()):
        prior = priors.get(label)
        if not prior:
            continue
        penalty = normalize_score(prior.get("penalty_factor", 0.35))
        reward = max(1.0, float(prior.get("reward_factor", 1.0)))
        if z_value is not None:
            if "min_z" in prior and z_value < float(prior["min_z"]):
                score *= penalty
            if "max_z" in prior and z_value > float(prior["max_z"]):
                score *= penalty
        if "min_mask_area_frac" in prior and area_frac < float(prior["min_mask_area_frac"]):
            score *= penalty
        if "max_mask_area_frac" in prior and area_frac > float(prior["max_mask_area_frac"]):
            score *= penalty
        if z_value is not None:
            z_ok = ("min_z" not in prior or z_value >= float(prior["min_z"])) and (
                "max_z" not in prior or z_value <= float(prior["max_z"])
            )
            area_ok = ("min_mask_area_frac" not in prior or area_frac >= float(prior["min_mask_area_frac"])) and (
                "max_mask_area_frac" not in prior or area_frac <= float(prior["max_mask_area_frac"])
            )
            if z_ok and area_ok:
                score *= reward
        adjusted[label] = normalize_score(score)
    return adjusted
