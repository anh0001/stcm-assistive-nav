#!/usr/bin/env python3

"""Unit tests for label calibration and cross-label instance voting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
PKG_ROOT = TEST_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from stcm.core import gng_instance_manager as gim
from stcm.core.gng_instance_manager import GngInstanceManager
from stcm.core.label_calibration import apply_geometry_priors, choose_label, combine_label_scores


class _FakeGngConfig:
    dim = 3


class _FakeGngModel:
    def __init__(self, config) -> None:
        self.config = config

    def insert(self, positions, probabilities=None) -> None:
        return None

    def pause(self) -> None:
        return None

    def nodes(self):
        return []

    def predict(self, centroid):
        return 0

    def run(self) -> None:
        return None

    def terminate(self) -> None:
        return None


def test_rerank_prefers_contextually_supported_label() -> None:
    combined = combine_label_scores(
        detector_label="bottle",
        detector_score=0.2,
        crop_scores={"bottle": 0.3, "cup": 0.95},
        image_priors={"bottle": 0.1, "cup": 0.8},
    )
    decision = choose_label(combined, min_margin=0.05)
    assert decision.label == "cup"
    assert decision.margin > 0.05


def test_rerank_rejects_low_margin_confusion() -> None:
    combined = combine_label_scores(
        detector_label="chair",
        detector_score=0.35,
        crop_scores={"chair": 0.55, "meeting table set": 0.58},
        image_priors={"chair": 0.42, "meeting table set": 0.45},
    )
    decision = choose_label(combined, min_margin=0.1)
    assert decision.label is None
    assert decision.margin < 0.1


def test_geometry_priors_downrank_implausible_label() -> None:
    adjusted = apply_geometry_priors(
        label_scores={"emergency exit sign": 0.9, "trash bin": 0.8},
        pose=np.array([10.5, 0.2, 0.12]),
        mask=np.ones((1, 40, 80), dtype=bool),
        priors={
            "emergency exit sign": {"min_z": 1.2, "max_mask_area_frac": 0.03, "penalty_factor": 0.1},
            "trash bin": {"max_z": 0.9, "penalty_factor": 0.35},
        },
    )
    decision = choose_label(adjusted, min_margin=0.01)
    assert decision.label == "trash bin"


def test_cross_label_instance_voting_reuses_track(monkeypatch) -> None:
    monkeypatch.setattr(gim, "GNGConfiguration", _FakeGngConfig)
    monkeypatch.setattr(gim, "GrowingNeuralGas", _FakeGngModel)

    manager = GngInstanceManager(
        enabled=True,
        per_label=True,
        max_nodes=100,
        lambda_=10,
        max_age=50,
        eps_w=0.05,
        eps_n=0.0006,
        alpha=0.95,
        beta=0.9995,
        min_observations_to_commit=1,
        cluster_merge_distance=0.5,
        outlier_gate_meters=0.0,
        instance_label_voting_enabled=True,
        cross_label_merge_distance_m=0.6,
    )

    first = manager.update(
        "emergency exit sign",
        np.array([1.0, 2.0, 0.1]),
        0.9,
        label_scores={"emergency exit sign": 0.9, "trash bin": 0.1},
    )
    second = manager.update(
        "trash bin",
        np.array([1.1, 2.05, 0.12]),
        0.95,
        label_scores={"trash bin": 1.0, "emergency exit sign": 0.0},
    )

    assert first is not None
    assert second is not None
    assert second.instance_id == first.instance_id
    assert second.label == "trash bin"
    assert second.label_votes["trash bin"] > second.label_votes["emergency exit sign"]
