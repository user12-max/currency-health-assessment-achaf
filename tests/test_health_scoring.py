"""Unit tests for Modules 11 and 12: AMFM Feature Fusion and Currency Health Index (CHI) Scoring."""

from __future__ import annotations

import numpy as np
import pytest

from src.health_scoring import assign_chi_category, compute_chi_from_scores


@pytest.mark.parametrize(
    "score,expected_category",
    [
        (100.0, "Excellent"),
        (90.0, "Excellent"),
        (89.99, "Good"),  # Must NOT round before category assignment
        (89.0, "Good"),
        (75.0, "Good"),
        (74.99, "Fair"),
        (74.0, "Fair"),
        (60.0, "Fair"),
        (59.99, "Poor"),
        (59.0, "Poor"),
        (40.0, "Poor"),
        (39.99, "Critical"),
        (39.0, "Critical"),
        (0.0, "Critical"),
    ],
)
def test_chi_boundary_cases(score: float, expected_category: str) -> None:
    """Verify exact CHI category boundaries (100, 90, 89, 75, 74, 60, 59, 40, 39, 0)."""
    assert assign_chi_category(score) == expected_category

    # Also test via full compute_chi_from_scores with uniform component scores
    comp_scores = {
        "edge": score,
        "fold": score,
        "texture": score,
        "color": score,
        "corner": score,
    }
    res = compute_chi_from_scores(comp_scores)
    assert res.chi_score is not None
    assert np.isclose(res.chi_score, score)
    assert res.health_category == expected_category


def test_chi_weighted_formula_calculation() -> None:
    """Verify CHI = 0.30*S_edge + 0.25*S_fold + 0.20*S_texture + 0.15*S_color + 0.10*S_corner."""
    scores = {
        "edge": 80.0,     # 0.30 * 80 = 24.0
        "fold": 60.0,     # 0.25 * 60 = 15.0
        "texture": 90.0,  # 0.20 * 90 = 18.0
        "color": 70.0,    # 0.15 * 70 = 10.5
        "corner": 50.0,   # 0.10 * 50 = 5.0
    }
    # Expected sum = 24.0 + 15.0 + 18.0 + 10.5 + 5.0 = 72.5 -> Fair
    res = compute_chi_from_scores(scores)
    assert res.chi_score is not None
    assert np.isclose(res.chi_score, 72.5)
    assert res.health_category == "Fair"
    assert res.scoring_approach == "Full 5-Component Weighted Fusion"
    assert res.is_provisional is False


def test_missing_feature_renormalization_and_withholding() -> None:
    """Verify missing features re-normalize weights when >=50% weight available, or withhold when <50%."""
    # Case 1: Texture (0.20) and Fold (0.25) missing -> available weight = 0.55 (>= 0.50)
    partial_scores = {
        "edge": 90.0,    # w = 0.30
        "fold": None,
        "texture": None,
        "color": 80.0,   # w = 0.15
        "corner": 70.0,  # w = 0.10
    }
    res_partial = compute_chi_from_scores(partial_scores)
    expected_chi = (0.30 * 90.0 + 0.15 * 80.0 + 0.10 * 70.0) / 0.55
    assert res_partial.chi_score is not None
    assert np.isclose(res_partial.chi_score, expected_chi)
    assert res_partial.scoring_approach == "Normalized Partial-Feature Weighted Fusion"
    assert res_partial.is_provisional is True
    assert np.isclose(sum(res_partial.effective_weights.values()), 1.0)

    # Case 2: Edge (0.30), Fold (0.25), and Texture (0.20) missing -> available weight = 0.25 (< 0.50)
    insufficient_scores = {
        "edge": None,
        "fold": None,
        "texture": None,
        "color": 85.0,
        "corner": 90.0,
    }
    res_withheld = compute_chi_from_scores(insufficient_scores)
    assert res_withheld.chi_score is None
    assert res_withheld.health_category == "Unable to Assess"
    assert "Withheld" in res_withheld.scoring_approach
