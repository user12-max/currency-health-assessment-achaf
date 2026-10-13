"""Module 11: Adaptive Multi-Feature Fusion Module (AMFM).

Combines complementary visual indicators from:
1. Edge integrity analysis (`edge`)
2. Fold/crease condition analysis (`fold`)
3. Surface texture analysis (`texture`)
4. Color and stain analysis (`color`)
5. Corner integrity analysis (`corner`)

into standardized component condition scores on a [0, 100] scale (where 100
represents the best estimated physical condition).

Key Design Principles:
- Directionality: Raw defect severity in [0, 100] (higher = worse) is inverted
  to Condition Score = 100 - AdjustedSeverity (higher = healthier).
- Capture-Quality Decoupling: Poor photographic capture quality (e.g., severe
  optical blur or extreme exposure clipping) is never silently converted into a
  physical damage penalty. If an indicator cannot be assessed reliably, its
  `is_available` flag is set to `False` and tracked explicitly.
- Adaptive Calibration: Minor false-positive noise floors induced by low
  resolution or uneven lighting are suppressed using capture-quality metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.color_analysis import ColorAnalysisResult
from src.damage_detection import DamageIndicatorResult
from src.edge_analysis import EdgeAnalysisResult
from src.image_quality import CaptureQualityReport
from src.morphology_analysis import MorphologyAnalysisResult
from src.texture_analysis import TextureAnalysisResult


@dataclass
class ComponentCondition:
    """Normalized condition score and metadata for a single AMFM feature channel."""

    component_key: str           # "edge", "fold", "texture", "color", "corner"
    display_name: str
    is_available: bool
    condition_score: Optional[float]  # [0.0, 100.0] when available, None if unavailable
    raw_defect_severity: float        # [0.0, 100.0] before quality adaptation
    reliability_label: str            # "High", "Moderate", "Low", or "Unavailable"
    used_reference: bool
    direction_note: str
    normalization_note: str
    assumptions_and_limitations: str
    status_explanation: str

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-serializable dictionary of this component's condition."""
        return {
            "component_key": self.component_key,
            "display_name": self.display_name,
            "is_available": self.is_available,
            "condition_score": (
                round(self.condition_score, 2)
                if self.condition_score is not None
                else None
            ),
            "raw_defect_severity": round(self.raw_defect_severity, 2),
            "reliability_label": self.reliability_label,
            "used_reference": self.used_reference,
            "direction_note": self.direction_note,
            "normalization_note": self.normalization_note,
            "assumptions_and_limitations": self.assumptions_and_limitations,
            "status_explanation": self.status_explanation,
        }


@dataclass
class AMFMFusionResult:
    """Complete output of the Adaptive Multi-Feature Fusion Module."""

    components: Dict[str, ComponentCondition]
    available_keys: List[str]
    unavailable_keys: List[str]
    is_provisional_capture: bool
    used_reference_mode: bool
    adaptation_notes: List[str] = field(default_factory=list)

    def get_score_map(self) -> Dict[str, Optional[float]]:
        """Return mapping of component_key -> condition_score (or None if unavailable)."""
        return {k: v.condition_score for k, v in self.components.items()}


def fuse_multi_features(
    edge_result: EdgeAnalysisResult,
    morph_result: MorphologyAnalysisResult,
    texture_result: TextureAnalysisResult,
    color_result: ColorAnalysisResult,
    indicators: Dict[str, DamageIndicatorResult],
    quality_report: CaptureQualityReport,
    segmentation_succeeded: bool = True,
) -> AMFMFusionResult:
    """Run the Adaptive Multi-Feature Fusion Module (AMFM).

    Parameters
    ----------
    edge_result : EdgeAnalysisResult
        Output from Module 6.
    morph_result : MorphologyAnalysisResult
        Output from Module 7.
    texture_result : TextureAnalysisResult
        Output from Module 5.
    color_result : ColorAnalysisResult
        Output from Module 8.
    indicators : dict of str -> DamageIndicatorResult
        Output from Module 10.
    quality_report : CaptureQualityReport
        Capture-quality metrics from Module 1/2.
    segmentation_succeeded : bool
        Whether automatic note boundary segmentation succeeded.

    Returns
    -------
    AMFMFusionResult
        Normalized 0-100 component condition scores, availability flags, and
        explainable adaptation notes.
    """
    adaptation_notes: List[str] = []

    # 1. Adaptive noise-floor damping based on image resolution and illumination
    illum_noise_allowance = 0.0
    if quality_report.illumination_nonuniformity > 0.22:
        illum_noise_allowance = min(
            4.5, (quality_report.illumination_nonuniformity - 0.22) * 18.0
        )
        adaptation_notes.append(
            f"Applied illumination-gradient noise allowance ({illum_noise_allowance:.1f} pts) "
            "to color and texture channels to avoid penalizing ambient lighting shadows."
        )

    low_res_allowance = 0.0
    if quality_report.is_low_resolution:
        low_res_allowance = 2.5
        adaptation_notes.append(
            "Applied low-resolution quantization allowance (2.5 pts) to boundary roughness "
            "and micro-texture measurements."
        )

    # --- Component 1: Edge Integrity (S_edge) ---
    edge_ind = indicators.get("edge_wear")
    edge_avail = edge_ind.is_assessed if edge_ind is not None else True
    raw_edge_sev = float(edge_result.overall_edge_wear_score)
    adj_edge_sev = max(0.0, raw_edge_sev - low_res_allowance * 0.6)
    edge_score = (
        float(np.clip(100.0 - adj_edge_sev, 0.0, 100.0)) if edge_avail else None
    )
    edge_rel = (
        edge_ind.confidence_label
        if (edge_ind is not None and edge_avail)
        else "Unavailable"
    )

    comp_edge = ComponentCondition(
        component_key="edge",
        display_name="Edge Integrity",
        is_available=edge_avail,
        condition_score=edge_score,
        raw_defect_severity=raw_edge_sev,
        reliability_label=edge_rel,
        used_reference=False,
        direction_note="0 = severely torn/eroded edges; 100 = continuous intact rectangular boundary.",
        normalization_note="Normalized by edge band area and length across top, bottom, left, and right borders.",
        assumptions_and_limitations=(
            "Assumes external note boundary is visible and contrasted against background. "
            "If manual cropping clips into the note or leaves background margin, accuracy decreases."
        ),
        status_explanation=edge_ind.explanation if edge_ind else edge_result.explanation,
    )

    # --- Component 2: Fold / Crease Condition (S_fold) ---
    fold_ind = indicators.get("fold_lines")
    fold_avail = fold_ind.is_assessed if fold_ind is not None else True
    raw_fold_sev = float(morph_result.fold_severity_score)
    adj_fold_sev = max(0.0, raw_fold_sev - (illum_noise_allowance * 0.4))
    fold_score = (
        float(np.clip(100.0 - adj_fold_sev, 0.0, 100.0)) if fold_avail else None
    )
    fold_rel = (
        fold_ind.confidence_label
        if (fold_ind is not None and fold_avail)
        else "Unavailable"
    )
    if not fold_avail:
        adaptation_notes.append(
            "Fold/crease component marked unavailable due to severe image blur (prevents "
            "mistaking optical blur for physical note condition)."
        )

    comp_fold = ComponentCondition(
        component_key="fold",
        display_name="Fold & Crease Condition",
        is_available=fold_avail,
        condition_score=fold_score,
        raw_defect_severity=raw_fold_sev,
        reliability_label=fold_rel,
        used_reference=morph_result.used_reference,
        direction_note="0 = heavy elongated creases or structural holes; 100 = flat crease-free surface.",
        normalization_note="Normalized by note height (segment length ratio) and interior area fraction.",
        assumptions_and_limitations=(
            "Detects visible shadow/ridge troughs from folds. Pressed flat creases under "
            "diffuse frontal lighting may have weak optical contrast."
        ),
        status_explanation=fold_ind.explanation if fold_ind else morph_result.explanation,
    )

    # --- Component 3: Surface Texture Condition (S_texture) ---
    tex_ind = indicators.get("texture_irregularity")
    tex_avail = tex_ind.is_assessed if tex_ind is not None else True
    raw_tex_sev = float(texture_result.texture_irregularity_score)
    adj_tex_sev = max(
        0.0, raw_tex_sev - (illum_noise_allowance * 0.5 + low_res_allowance * 0.5)
    )
    tex_score = (
        float(np.clip(100.0 - adj_tex_sev, 0.0, 100.0)) if tex_avail else None
    )
    tex_rel = (
        tex_ind.confidence_label
        if (tex_ind is not None and tex_avail)
        else "Unavailable"
    )
    if not tex_avail:
        adaptation_notes.append(
            "Surface texture component marked unavailable due to severe optical blur."
        )

    comp_texture = ComponentCondition(
        component_key="texture",
        display_name="Surface Texture Condition",
        is_available=tex_avail,
        condition_score=tex_score,
        raw_defect_severity=raw_tex_sev,
        reliability_label=tex_rel,
        used_reference=texture_result.used_reference,
        direction_note="0 = severe crumple/scuff texture anomaly; 100 = clean surface texture.",
        normalization_note="Patch-wise GLCM homogeneity and uniform LBP entropy normalized across a 4x8 grid.",
        assumptions_and_limitations=(
            "Single-image mode isolates high-frequency isotropic roughness from intentional "
            "print edges; reference-assisted mode compares homologous regions directly."
        ),
        status_explanation=tex_ind.explanation if tex_ind else texture_result.explanation,
    )

    # --- Component 4: Stain & Color Condition (S_color) ---
    stain_ind = indicators.get("stain_regions")
    fading_ind = indicators.get("fading_discoloration")
    stain_avail = stain_ind.is_assessed if stain_ind is not None else True
    fading_avail = fading_ind.is_assessed if fading_ind is not None else True
    color_avail = bool(stain_avail or fading_avail)

    if stain_avail and fading_avail:
        raw_color_sev = (
            0.72 * color_result.stain_anomaly_score
            + 0.28 * color_result.fading_discoloration_score
        )
    elif stain_avail:
        raw_color_sev = float(color_result.stain_anomaly_score)
    elif fading_avail:
        raw_color_sev = float(color_result.fading_discoloration_score)
    else:
        raw_color_sev = 0.0

    adj_color_sev = max(0.0, raw_color_sev - illum_noise_allowance)
    color_score = (
        float(np.clip(100.0 - adj_color_sev, 0.0, 100.0)) if color_avail else None
    )
    color_rel = (
        stain_ind.confidence_label
        if (stain_ind is not None and stain_avail)
        else "Unavailable"
    )

    comp_color = ComponentCondition(
        component_key="color",
        display_name="Stain & Color Condition",
        is_available=color_avail,
        condition_score=color_score,
        raw_defect_severity=raw_color_sev,
        reliability_label=color_rel,
        used_reference=color_result.used_reference,
        direction_note="0 = heavy localized staining or severe discoloration; 100 = clean unblemished color.",
        normalization_note="Normalized in CIE Lab and HSV spaces after L-channel illumination leveling.",
        assumptions_and_limitations=(
            "Distinguishes localized blotches from broad lighting gradients. Global ink "
            "fading assessment is most reliable when an aligned reference note is supplied."
        ),
        status_explanation=color_result.explanation,
    )

    # --- Component 5: Corner Integrity (S_corner) ---
    corner_ind = indicators.get("corner_damage")
    corner_avail = corner_ind.is_assessed if corner_ind is not None else True
    raw_corner_sev = float(edge_result.overall_corner_damage_score)
    adj_corner_sev = max(0.0, raw_corner_sev - low_res_allowance * 0.5)
    corner_score = (
        float(np.clip(100.0 - adj_corner_sev, 0.0, 100.0)) if corner_avail else None
    )
    corner_rel = (
        corner_ind.confidence_label
        if (corner_ind is not None and corner_avail)
        else "Unavailable"
    )

    comp_corner = ComponentCondition(
        component_key="corner",
        display_name="Corner Integrity",
        is_available=corner_avail,
        condition_score=corner_score,
        raw_defect_severity=raw_corner_sev,
        reliability_label=corner_rel,
        used_reference=False,
        direction_note="0 = severely torn/missing corners; 100 = all four corners sharp and intact.",
        normalization_note="Normalized by corner box area retention and vertex tip distance across 4 corners.",
        assumptions_and_limitations=(
            "Evaluates foreground retention within the four normalized 16% corner boxes "
            "of the rectified quadrilateral envelope."
        ),
        status_explanation=corner_ind.explanation if corner_ind else edge_result.explanation,
    )

    components: Dict[str, ComponentCondition] = {
        "edge": comp_edge,
        "fold": comp_fold,
        "texture": comp_texture,
        "color": comp_color,
        "corner": comp_corner,
    }

    available_keys = [k for k, v in components.items() if v.is_available and v.condition_score is not None]
    unavailable_keys = [k for k in components.keys() if k not in available_keys]
    used_ref_any = any(v.used_reference for v in components.values())

    return AMFMFusionResult(
        components=components,
        available_keys=available_keys,
        unavailable_keys=unavailable_keys,
        is_provisional_capture=quality_report.is_provisional,
        used_reference_mode=used_ref_any,
        adaptation_notes=adaptation_notes,
    )
