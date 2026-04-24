#!/usr/bin/env python3

"""Unit tests for frozen supervised semantic-prior fusion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

TEST_DIR = Path(__file__).resolve().parent
PKG_ROOT = TEST_DIR.parent
REPO_ROOT = PKG_ROOT.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from stcm.core.supervised_semantic_prior import (  # noqa: E402
    SemanticPriorPrediction,
    canonical_prior_map,
    extract_prior_candidates,
    fuse_label_scores_with_semantic_prior,
)


def _canonicalize(phrase: str) -> str | None:
    lookup = {
        "chair": "chair",
        "chairs": "chair",
        "meeting table": "meeting table set",
        "meeting table set": "meeting table set",
        "bottle": "bottle",
    }
    return lookup.get(str(phrase).lower())


def test_canonical_prior_map_limits_to_active_target_labels() -> None:
    mapped = canonical_prior_map(["chair", "meeting table set", "bottle"])
    assert mapped[5] == ("chair",)
    assert mapped[7] == ("meeting table set",)
    assert 29 not in mapped


def test_semantic_prior_boosts_agreeing_mask_score() -> None:
    label_map = np.zeros((4, 4), dtype=np.uint8)
    label_map[:, :] = 5
    masks = torch.ones((1, 1, 4, 4), dtype=torch.bool)
    scores = torch.tensor([0.5], dtype=torch.float32)

    adjusted, label_maps = fuse_label_scores_with_semantic_prior(
        phrases=["chair"],
        scores=scores,
        masks=masks,
        prediction=SemanticPriorPrediction(label_map=label_map, backend="test"),
        target_labels=["chair"],
        canonicalize=_canonicalize,
    )

    assert float(adjusted[0]) > 0.5
    assert label_maps is not None
    assert label_maps[0]["chair"] > 0.5


def test_semantic_prior_penalizes_disagreeing_mapped_label() -> None:
    label_map = np.zeros((4, 4), dtype=np.uint8)
    label_map[:, :] = 7
    masks = torch.ones((1, 1, 4, 4), dtype=torch.bool)
    scores = torch.tensor([0.8], dtype=torch.float32)

    adjusted, label_maps = fuse_label_scores_with_semantic_prior(
        phrases=["chair"],
        scores=scores,
        masks=masks,
        prediction=SemanticPriorPrediction(label_map=label_map, backend="test"),
        target_labels=["chair", "meeting table set"],
        canonicalize=_canonicalize,
    )

    assert float(adjusted[0]) < 0.8
    assert label_maps is not None
    assert label_maps[0]["meeting table set"] == 1.0


def test_semantic_prior_noops_for_unmapped_target_label() -> None:
    label_map = np.full((4, 4), 5, dtype=np.uint8)
    masks = torch.ones((1, 1, 4, 4), dtype=torch.bool)
    scores = torch.tensor([0.6], dtype=torch.float32)

    adjusted, label_maps = fuse_label_scores_with_semantic_prior(
        phrases=["bottle"],
        scores=scores,
        masks=masks,
        prediction=SemanticPriorPrediction(label_map=label_map, backend="test"),
        target_labels=["bottle"],
        canonicalize=_canonicalize,
    )

    assert float(adjusted[0]) == float(scores[0])
    assert label_maps is None


def test_extract_prior_candidates_skips_existing_labels_and_huge_components() -> None:
    label_map = np.zeros((20, 20), dtype=np.uint8)
    label_map[1:19, 1:19] = 8
    label_map[2:8, 2:8] = 5
    label_map[10:15, 10:15] = 29

    candidates = extract_prior_candidates(
        prediction=SemanticPriorPrediction(label_map=label_map, backend="test"),
        target_labels=["chair", "door", "cardboard box"],
        existing_labels=["chair"],
        min_area_px=10,
        max_area_frac=0.5,
        max_per_label=2,
        score=0.4,
    )

    labels = [item.label for item in candidates]
    assert "chair" not in labels
    assert "door" not in labels
    assert labels == ["cardboard box"]
    assert candidates[0].box_xyxy == (10.0, 10.0, 15.0, 15.0)


def test_semantic_prior_config_defaults_off() -> None:
    with open(REPO_ROOT / "stcm" / "config" / "semantic_mapping_params.yaml") as handle:
        cfg = yaml.safe_load(handle)
    assert cfg["semantic_prior_backend"] == "none"
    assert cfg["semantic_prior_fusion_enabled"] is True
