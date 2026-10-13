"""Module 20: Results and Diagnostic Visualizations.

Creates clean, publication-grade Matplotlib figures for the Streamlit research
dashboard:
- Component condition score bar chart with CHI threshold reference lines.
- Currency Health Index (CHI) horizontal gauge bar.
- Multi-panel diagnostic masks and feature maps figure.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.feature_fusion import AMFMFusionResult
from src.health_scoring import CHIScoringResult

# Use non-interactive Agg backend for thread safety in Streamlit and pytest
matplotlib.use("Agg")


def _category_color_hex(score: Optional[float]) -> str:
    """Return a consistent semantic hex color for a 0-100 condition score."""
    if score is None or not np.isfinite(score):
        return "#94A3B8"  # Slate gray for unavailable
    if score >= 90.0:
        return "#16A34A"  # Emerald green (Excellent)
    if score >= 75.0:
        return "#2563EB"  # Royal blue (Good)
    if score >= 60.0:
        return "#D97706"  # Amber (Fair)
    if score >= 40.0:
        return "#EA580C"  # Orange (Poor)
    return "#DC2626"      # Crimson red (Critical)


def create_component_scores_chart(
    amfm_result: AMFMFusionResult,
    chi_result: CHIScoringResult,
) -> plt.Figure:
    """Create a horizontal bar chart of the 5 AMFM component condition scores."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#F8FAFC")

    order = ["edge", "fold", "texture", "color", "corner"]
    labels = []
    scores = []
    bar_colors = []

    for k in reversed(order):
        comp = amfm_result.components[k]
        w_eff = chi_result.effective_weights.get(k, 0.0)
        labels.append(f"{comp.display_name}\n(w={w_eff:.2f})")
        if comp.is_available and comp.condition_score is not None:
            scores.append(float(comp.condition_score))
            bar_colors.append(_category_color_hex(comp.condition_score))
        else:
            scores.append(0.0)
            bar_colors.append("#CBD5E1")

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, height=0.58, color=bar_colors, edgecolor="#1E293B", linewidth=0.6)

    # Reference threshold lines
    for thresh, col, lbl in [
        (90, "#16A34A", "90"),
        (75, "#2563EB", "75"),
        (60, "#D97706", "60"),
        (40, "#DC2626", "40"),
    ]:
        ax.axvline(thresh, color=col, linestyle="--", linewidth=0.8, alpha=0.55)

    for idx, bar in enumerate(bars):
        comp_key = list(reversed(order))[idx]
        comp = amfm_result.components[comp_key]
        if comp.is_available and comp.condition_score is not None:
            txt = f" {comp.condition_score:.1f} / 100 ({comp.reliability_label})"
        else:
            txt = " Not reliably assessed"
        ax.text(
            min(bar.get_width() + 1.5, 76.0),
            bar.get_y() + bar.get_height() / 2.0,
            txt,
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#0F172A",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5, color="#0F172A")
    ax.set_xlim(0, 112)
    ax.set_xlabel("Condition Score (0 = Severe Degradation, 100 = Intact)", fontsize=8.5, color="#334155")
    ax.set_title("AMFM Component Condition Scores & Effective Weights", fontsize=10, fontweight="bold", color="#0F172A", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def create_chi_gauge_chart(chi_result: CHIScoringResult) -> plt.Figure:
    """Create a horizontal segmented spectrum gauge indicating the overall CHI score."""
    fig, ax = plt.subplots(figsize=(7.2, 1.55), dpi=130)
    fig.patch.set_facecolor("#FFFFFF")

    bands = [
        (0, 40, "#FEE2E2", "#DC2626", "Critical (<40)"),
        (40, 60, "#FFEDD5", "#EA580C", "Poor (40-59)"),
        (60, 75, "#FEF3C7", "#D97706", "Fair (60-74)"),
        (75, 90, "#DBEAFE", "#2563EB", "Good (75-89)"),
        (90, 100, "#DCFCE7", "#16A34A", "Excellent (90-100)"),
    ]

    for start, end, bg_hex, fg_hex, b_label in bands:
        ax.barh(0, end - start, left=start, height=0.55, color=bg_hex, edgecolor=fg_hex, linewidth=1.0)
        mid = (start + end) / 2.0
        ax.text(mid, 0, b_label, ha="center", va="center", fontsize=7.5, fontweight="bold", color=fg_hex)

    if chi_result.chi_score is not None:
        val = float(np.clip(chi_result.chi_score, 0.0, 100.0))
        marker_col = _category_color_hex(val)
        ax.axvline(val, color="#0F172A", linewidth=2.5, zorder=5)
        ax.scatter([val], [0.32], color=marker_col, edgecolor="#0F172A", s=70, zorder=6, marker="v")
        ax.set_title(
            f"Currency Health Index (CHI): {val:.1f} / 100 — {chi_result.health_category}",
            fontsize=10,
            fontweight="bold",
            color="#0F172A",
            pad=6,
        )
    else:
        ax.set_title(
            "Currency Health Index (CHI): Withheld (Insufficient Reliable Evidence)",
            fontsize=10,
            fontweight="bold",
            color="#64748B",
            pad=6,
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.4, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("CHI Scale (0–100)", fontsize=8, color="#475569")
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
