"""Module 12: Currency Health Index (CHI) Scoring and Categorization.

Computes the project-defined Currency Health Index (CHI) from the five AMFM
component condition scores:
    CHI = 0.30 * S_edge
        + 0.25 * S_fold
        + 0.20 * S_texture
        + 0.15 * S_color
        + 0.10 * S_corner

Features:
- Configurable weights (centrally defined in `config/settings.py`).
- Explicit handling of missing/unavailable components:
  * Never substitutes arbitrary default scores (such as 100 or 0) for missing
    components.
  * Re-normalizes weights over available components when total available weight
    >= `MIN_AVAILABLE_WEIGHT_FOR_CHI` (0.50), clearly reporting the re-normalized
    formula.
  * Withholds the overall CHI score (`chi_score = None`) when available evidence
    falls below `MIN_AVAILABLE_WEIGHT_FOR_CHI`.
- Strict unrounded boundary categorization:
  * 90 <= CHI <= 100 : Excellent
  * 75 <= CHI < 90   : Good
  * 60 <= CHI < 75   : Fair
  * 40 <= CHI < 60   : Poor
  * CHI < 40         : Critical
  Rounding is performed ONLY for display formatting, never prior to category
  assignment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config.settings import (
    DEFAULT_CHI_WEIGHTS,
    MIN_AVAILABLE_WEIGHT_FOR_CHI,
    PROTOTYPE_DISCLAIMER,
)
from src.feature_fusion import AMFMFusionResult


@dataclass
class CHIScoringResult:
    """Structured result of the Currency Health Index (CHI) calculation."""

    chi_score: Optional[float]            # Exact unrounded score in [0, 100], or None if withheld
    chi_display: str                      # Formatted string, e.g., "84.5 / 100" or "Withheld"
    health_category: str                  # "Excellent", "Good", "Fair", "Poor", "Critical", or "Unable to Assess"
    is_provisional: bool                  # True if capture quality is poor or partial features used
    scoring_approach: str                 # "Full 5-Component Weighted Fusion", "Normalized Partial-Feature Weighted Fusion", or "Withheld (Insufficient Evidence)"
    configured_weights: Dict[str, float]
    effective_weights: Dict[str, float]
    component_scores: Dict[str, Optional[float]]
    weighted_contributions: Dict[str, float]
    available_weight_fraction: float
    limitations_and_notes: List[str] = field(default_factory=list)
    disclaimer: str = PROTOTYPE_DISCLAIMER

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-serializable dictionary of CHI results."""
        return {
            "chi_score": round(self.chi_score, 4) if self.chi_score is not None else None,
            "chi_score_rounded": round(self.chi_score, 1) if self.chi_score is not None else None,
            "chi_display": self.chi_display,
            "health_category": self.health_category,
            "is_provisional": self.is_provisional,
            "scoring_approach": self.scoring_approach,
            "configured_weights": {k: round(v, 4) for k, v in self.configured_weights.items()},
            "effective_weights": {k: round(v, 4) for k, v in self.effective_weights.items()},
            "component_scores": {
                k: (round(v, 2) if v is not None else None)
                for k, v in self.component_scores.items()
            },
            "weighted_contributions": {
                k: round(v, 2) for k, v in self.weighted_contributions.items()
            },
            "available_weight_fraction": round(self.available_weight_fraction, 4),
            "limitations_and_notes": list(self.limitations_and_notes),
            "disclaimer": self.disclaimer,
        }


def assign_chi_category(chi_score: Optional[float]) -> str:
    """Assign a health category from an unrounded CHI score in [0, 100].

    Boundary rules (evaluated without prior rounding):
    - CHI >= 90.0            -> 'Excellent'
    - 75.0 <= CHI < 90.0     -> 'Good'
    - 60.0 <= CHI < 75.0     -> 'Fair'
    - 40.0 <= CHI < 60.0     -> 'Poor'
    - CHI < 40.0             -> 'Critical'
    """
    if chi_score is None or not np.isfinite(chi_score):
        return "Unable to Assess"

    val = float(chi_score)
    if val >= 90.0:
        return "Excellent"
    if val >= 75.0:
        return "Good"
    if val >= 60.0:
        return "Fair"
    if val >= 40.0:
        return "Poor"
    return "Critical"


def compute_chi_from_scores(
    component_scores: Dict[str, Optional[float]],
    weights: Optional[Dict[str, float]] = None,
    is_provisional_capture: bool = False,
    min_available_weight: float = MIN_AVAILABLE_WEIGHT_FOR_CHI,
) -> CHIScoringResult:
    """Compute CHI directly from a dictionary of component condition scores.

    Parameters
    ----------
    component_scores : dict of str -> Optional[float]
        Mapping with keys `'edge'`, `'fold'`, `'texture'`, `'color'`, `'corner'`.
        Values must be in `[0, 100]` or `None` (when a component could not be
        reliably assessed).
    weights : dict of str -> float, optional
        Component weights. Defaults to `DEFAULT_CHI_WEIGHTS`
        (edge=0.30, fold=0.25, texture=0.20, color=0.15, corner=0.10).
    is_provisional_capture : bool
        Whether poor image capture quality marks the result as provisional.
    min_available_weight : float
        Minimum fraction of total configured weight that must be available to
        compute an overall CHI score.

    Returns
    -------
    CHIScoringResult
        Calculated CHI, category, effective weights, and diagnostic notes.
    """
    raw_weights = dict(weights) if weights is not None else dict(DEFAULT_CHI_WEIGHTS)
    expected_keys = ["edge", "fold", "texture", "color", "corner"]

    # Normalize configured weights so they sum to 1.0
    total_cfg_weight = sum(max(0.0, float(raw_weights.get(k, 0.0))) for k in expected_keys)
    if total_cfg_weight <= 1e-9:
        configured_weights = dict(DEFAULT_CHI_WEIGHTS)
    else:
        configured_weights = {
            k: max(0.0, float(raw_weights.get(k, 0.0))) / total_cfg_weight
            for k in expected_keys
        }

    clean_scores: Dict[str, Optional[float]] = {}
    available_keys: List[str] = []
    missing_keys: List[str] = []

    for k in expected_keys:
        val = component_scores.get(k)
        if val is not None and np.isfinite(val):
            clean_scores[k] = float(np.clip(val, 0.0, 100.0))
            available_keys.append(k)
        else:
            clean_scores[k] = None
            missing_keys.append(k)

    available_weight_sum = sum(configured_weights[k] for k in available_keys)
    notes: List[str] = []

    if available_weight_sum < min_available_weight or len(available_keys) == 0:
        notes.append(
            f"Overall Currency Health Index (CHI) withheld: available components "
            f"({', '.join(available_keys) if available_keys else 'none'}) represent only "
            f"{available_weight_sum * 100:.1f}% of total weight (minimum required: "
            f"{min_available_weight * 100:.0f}%)."
        )
        return CHIScoringResult(
            chi_score=None,
            chi_display="Withheld (Insufficient Evidence)",
            health_category="Unable to Assess",
            is_provisional=True,
            scoring_approach="Withheld (Insufficient Available Feature Weight)",
            configured_weights=configured_weights,
            effective_weights={k: 0.0 for k in expected_keys},
            component_scores=clean_scores,
            weighted_contributions={k: 0.0 for k in expected_keys},
            available_weight_fraction=available_weight_sum,
            limitations_and_notes=notes,
        )

    # Re-normalize weights across available components (equals configured_weights when all 5 are available)
    effective_weights: Dict[str, float] = {}
    weighted_contributions: Dict[str, float] = {}
    chi_accum = 0.0

    for k in expected_keys:
        if k in available_keys and clean_scores[k] is not None:
            eff_w = configured_weights[k] / available_weight_sum
            effective_weights[k] = eff_w
            contrib = eff_w * float(clean_scores[k])
            weighted_contributions[k] = contrib
            chi_accum += contrib
        else:
            effective_weights[k] = 0.0
            weighted_contributions[k] = 0.0

    # Clamp strictly within [0.0, 100.0] while keeping floating-point precision for boundary checks
    chi_unrounded = float(np.clip(chi_accum, 0.0, 100.0))
    category = assign_chi_category(chi_unrounded)

    if len(missing_keys) == 0:
        scoring_approach = "Full 5-Component Weighted Fusion"
    else:
        scoring_approach = "Normalized Partial-Feature Weighted Fusion"
        notes.append(
            f"Components unavailable ({', '.join(missing_keys)}); CHI was computed via "
            f"re-normalized weighted fusion over available components "
            f"({', '.join(available_keys)}, covering {available_weight_sum * 100:.1f}% of base weight)."
        )

    is_prov = bool(is_provisional_capture or len(missing_keys) > 0)
    if is_provisional_capture:
        notes.append(
            "Assessment is marked PROVISIONAL due to capture-quality limitations."
        )

    prov_suffix = " (Provisional)" if is_prov else ""
    chi_display = f"{chi_unrounded:.1f} / 100{prov_suffix}"

    return CHIScoringResult(
        chi_score=chi_unrounded,
        chi_display=chi_display,
        health_category=category,
        is_provisional=is_prov,
        scoring_approach=scoring_approach,
        configured_weights=configured_weights,
        effective_weights=effective_weights,
        component_scores=clean_scores,
        weighted_contributions=weighted_contributions,
        available_weight_fraction=available_weight_sum,
        limitations_and_notes=notes,
    )


def calculate_currency_health_index(
    amfm_result: AMFMFusionResult,
    weights: Optional[Dict[str, float]] = None,
) -> CHIScoringResult:
    """Calculate the Currency Health Index (CHI) from an `AMFMFusionResult`."""
    score_map = amfm_result.get_score_map()
    res = compute_chi_from_scores(
        component_scores=score_map,
        weights=weights,
        is_provisional_capture=amfm_result.is_provisional_capture,
    )
    for note in amfm_result.adaptation_notes:
        if note not in res.limitations_and_notes:
            res.limitations_and_notes.append(note)
    return res
