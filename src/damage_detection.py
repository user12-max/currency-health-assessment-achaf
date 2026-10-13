"""Module 10: Damage-Indicator Detection.

Provides six dedicated, explainable detector functions for physical deterioration
indicators on currency notes:
A. Edge wear (`detect_edge_wear`)
B. Corner damage (`detect_corner_damage`)
C. Fold-like lines (`detect_fold_lines`)
D. Surface texture irregularity (`detect_texture_irregularity`)
E. Stain-like regions (`detect_stain_regions`)
F. Possible fading or discoloration (`detect_fading_discoloration`)

Each detector returns a structured `DamageIndicatorResult` with qualitative
confidence labels ('High', 'Moderate', 'Low', or 'Not reliably assessed')
derived from explicit capture-quality, segmentation, and registration checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.color_analysis import ColorAnalysisResult
from src.edge_analysis import EdgeAnalysisResult
from src.image_quality import CaptureQualityReport
from src.morphology_analysis import MorphologyAnalysisResult
from src.region_analysis import NoteRegionMap
from src.texture_analysis import TextureAnalysisResult


@dataclass
class DamageIndicatorResult:
    """Structured output for a single damage-indicator category."""

    category_key: str
    category_name: str
    overlay_color_key: str  # "edge_corner", "fold_line", "stain_color", or "texture"
    is_assessed: bool
    is_detected: bool
    severity_score: float  # [0.0, 100.0]
    severity_label: str    # "None detected", "Minor", "Moderate", "Severe", "Not reliably assessed"
    confidence_label: str  # "High", "Moderate", "Low", "Not reliably assessed"
    used_reference: bool
    mask: np.ndarray       # uint8 binary mask (H, W) in rectified coordinates
    bounding_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (x, y, w, h)
    affected_locations: List[str] = field(default_factory=list)
    region_count: int = 0
    affected_area_pct: float = 0.0
    explanation: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-serializable summary of this damage indicator."""
        return {
            "category_key": self.category_key,
            "category_name": self.category_name,
            "overlay_color_key": self.overlay_color_key,
            "is_assessed": self.is_assessed,
            "is_detected": self.is_detected,
            "severity_score": round(self.severity_score, 2),
            "severity_label": self.severity_label,
            "confidence_label": self.confidence_label,
            "used_reference": self.used_reference,
            "region_count": self.region_count,
            "affected_area_pct": round(self.affected_area_pct, 3),
            "bounding_boxes": [list(b) for b in self.bounding_boxes],
            "affected_locations": list(self.affected_locations),
            "explanation": self.explanation,
        }


def _severity_label_from_score(score: float, is_assessed: bool, threshold: float = 6.0) -> Tuple[bool, str]:
    """Convert a 0-100 severity score into detection flag and qualitative severity label."""
    if not is_assessed:
        return False, "Not reliably assessed"
    if score < threshold:
        return False, "None detected"
    if score < 22.0:
        return True, "Minor"
    if score < 50.0:
        return True, "Moderate"
    return True, "Severe"


def _extract_bboxes_and_locations(
    binary_mask: np.ndarray,
    region_map: NoteRegionMap,
    min_area_px: int = 12,
) -> Tuple[List[Tuple[int, int, int, int]], List[str], float]:
    """Extract connected component bounding boxes, anatomical locations, and area percentage."""
    if binary_mask is None or binary_mask.size == 0 or not np.any(binary_mask > 0):
        return [], [], 0.0

    h, w = binary_mask.shape[:2]
    total_area = float(max(1, h * w))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8), connectivity=8
    )

    bboxes: List[Tuple[int, int, int, int]] = []
    locations: List[str] = []
    valid_pixel_count = 0

    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])

        bboxes.append((x, y, bw, bh))
        loc = region_map.locate_point_region(cx, cy)
        if loc not in locations:
            locations.append(loc)
        valid_pixel_count += area

    area_pct = (float(valid_pixel_count) / total_area) * 100.0
    return bboxes, locations, area_pct


def detect_edge_wear(
    edge_result: EdgeAnalysisResult,
    region_map: NoteRegionMap,
    quality_report: CaptureQualityReport,
    segmentation_succeeded: bool,
) -> DamageIndicatorResult:
    """A. Identify and quantify external boundary edge wear and irregularities."""
    score = edge_result.overall_edge_wear_score
    mask = edge_result.edge_wear_mask.copy()

    # Determine qualitative confidence based on segmentation and image sharpness
    if quality_report.is_severely_blurry and not segmentation_succeeded:
        is_assessed = False
        conf = "Not reliably assessed"
    elif not segmentation_succeeded:
        is_assessed = True
        conf = "Low"
    elif quality_report.is_blurry:
        is_assessed = True
        conf = "Moderate"
    else:
        is_assessed = True
        conf = "High"

    is_det, sev_label = _severity_label_from_score(score, is_assessed, threshold=6.0)
    if not is_det:
        mask = np.zeros_like(mask)

    bboxes, locs, area_pct = _extract_bboxes_and_locations(mask, region_map, min_area_px=10)
    affected_edges = [
        f"{name} ({m.edge_wear_score:.1f}%)"
        for name, m in edge_result.edge_metrics.items()
        if m.edge_wear_score >= 6.0
    ]

    if not is_assessed:
        expl = "Unable to reliably assess outer edge wear due to severe blur and unsegmented boundary."
    elif is_det:
        edge_str = ", ".join(affected_edges) if affected_edges else "outer perimeter"
        expl = f"Boundary irregularity or missing margin detected along: {edge_str}."
    else:
        expl = "All four external note boundaries appear continuous with minimal wear."

    return DamageIndicatorResult(
        category_key="edge_wear",
        category_name="Edge Wear & Boundary Irregularity",
        overlay_color_key="edge_corner",
        is_assessed=is_assessed,
        is_detected=is_det,
        severity_score=score if is_assessed else 0.0,
        severity_label=sev_label,
        confidence_label=conf,
        used_reference=False,
        mask=mask,
        bounding_boxes=bboxes,
        affected_locations=locs or ([f"Edge ({e.split()[0]})" for e in affected_edges] if is_det else []),
        region_count=len(bboxes),
        affected_area_pct=area_pct,
        explanation=expl,
    )


def detect_corner_damage(
    edge_result: EdgeAnalysisResult,
    region_map: NoteRegionMap,
    quality_report: CaptureQualityReport,
    segmentation_succeeded: bool,
) -> DamageIndicatorResult:
    """B. Identify and quantify missing, blunted, or folded note corners."""
    score = edge_result.overall_corner_damage_score
    mask = edge_result.corner_damage_mask.copy()

    if quality_report.is_severely_blurry and not segmentation_succeeded:
        is_assessed = False
        conf = "Not reliably assessed"
    elif not segmentation_succeeded:
        is_assessed = True
        conf = "Low"
    elif quality_report.is_blurry:
        is_assessed = True
        conf = "Moderate"
    else:
        is_assessed = True
        conf = "High"

    is_det, sev_label = _severity_label_from_score(score, is_assessed, threshold=6.0)
    if not is_det:
        mask = np.zeros_like(mask)

    bboxes, locs, area_pct = _extract_bboxes_and_locations(mask, region_map, min_area_px=10)
    affected_corners = [
        f"{name.replace('_', '-')} ({m.corner_damage_score:.1f}%)"
        for name, m in edge_result.corner_metrics.items()
        if m.corner_damage_score >= 6.0
    ]

    if not is_assessed:
        expl = "Unable to reliably assess corner geometry due to severe blur and unsegmented boundary."
    elif is_det:
        c_str = ", ".join(affected_corners) if affected_corners else "corner region"
        expl = f"Corner deficit or dog-ear distortion detected at: {c_str}."
    else:
        expl = "All four note corners retain expected rectangular geometry."

    return DamageIndicatorResult(
        category_key="corner_damage",
        category_name="Missing or Damaged Corners",
        overlay_color_key="edge_corner",
        is_assessed=is_assessed,
        is_detected=is_det,
        severity_score=score if is_assessed else 0.0,
        severity_label=sev_label,
        confidence_label=conf,
        used_reference=False,
        mask=mask,
        bounding_boxes=bboxes,
        affected_locations=locs or ([f"Corner ({c.split()[0]})" for c in affected_corners] if is_det else []),
        region_count=len(bboxes),
        affected_area_pct=area_pct,
        explanation=expl,
    )


def detect_fold_lines(
    morph_result: MorphologyAnalysisResult,
    region_map: NoteRegionMap,
    quality_report: CaptureQualityReport,
) -> DamageIndicatorResult:
    """C. Identify fold-like and crease-like linear anomalies and structural gaps."""
    score = morph_result.fold_severity_score
    mask = cv2.bitwise_or(
        morph_result.fold_candidate_mask, morph_result.structural_hole_mask
    )

    if quality_report.is_severely_blurry:
        is_assessed = False
        conf = "Not reliably assessed"
    elif quality_report.is_blurry or quality_report.is_low_resolution:
        is_assessed = True
        conf = "Low"
    elif morph_result.used_reference:
        is_assessed = True
        conf = "High"
    else:
        is_assessed = True
        conf = "Moderate"

    is_det, sev_label = _severity_label_from_score(score, is_assessed, threshold=7.0)
    if not is_det:
        mask = np.zeros_like(mask)

    bboxes, locs, area_pct = _extract_bboxes_and_locations(mask, region_map, min_area_px=18)
    n_lines = len(morph_result.detected_fold_lines)
    n_holes = morph_result.hole_count

    if not is_assessed:
        expl = (
            "Not reliably assessed: severe image blur prevents dependable detection of "
            "thin fold or crease lines."
        )
    elif is_det:
        expl = (
            f"Detected {n_lines} elongated fold/crease-like line indicator(s) "
            f"and {n_holes} interior structural gap(s) via directional morphological analysis."
        )
    else:
        expl = "No prominent elongated fold lines or interior structural gaps detected."

    return DamageIndicatorResult(
        category_key="fold_lines",
        category_name="Fold-like & Crease-like Lines",
        overlay_color_key="fold_line",
        is_assessed=is_assessed,
        is_detected=is_det,
        severity_score=score if is_assessed else 0.0,
        severity_label=sev_label,
        confidence_label=conf,
        used_reference=morph_result.used_reference,
        mask=mask,
        bounding_boxes=bboxes,
        affected_locations=locs,
        region_count=len(bboxes),
        affected_area_pct=area_pct,
        explanation=expl,
    )


def detect_texture_irregularity(
    texture_result: TextureAnalysisResult,
    region_map: NoteRegionMap,
    quality_report: CaptureQualityReport,
) -> DamageIndicatorResult:
    """D. Identify localized surface texture irregularities (crumpling/abrasion)."""
    score = texture_result.texture_irregularity_score
    mask = texture_result.anomaly_mask.copy()

    if quality_report.is_severely_blurry:
        is_assessed = False
        conf = "Not reliably assessed"
    elif quality_report.is_blurry or quality_report.is_low_resolution:
        is_assessed = True
        conf = "Low"
    elif texture_result.used_reference:
        is_assessed = True
        conf = "High"
    else:
        is_assessed = True
        conf = "Moderate"

    is_det, sev_label = _severity_label_from_score(score, is_assessed, threshold=8.0)
    if not is_det:
        mask = np.zeros_like(mask)

    bboxes, locs, area_pct = _extract_bboxes_and_locations(mask, region_map, min_area_px=25)

    if not is_assessed:
        expl = (
            "Not reliably assessed: severe optical blur masks fine-scale GLCM and LBP "
            "surface texture variations."
        )
    elif is_det:
        expl = (
            f"Localized surface roughness / texture anomaly detected across {len(bboxes)} "
            f"region(s) ({area_pct:.1f}% note area; GLCM homogeneity="
            f"{texture_result.global_glcm_homogeneity:.2f})."
        )
    else:
        expl = (
            f"Surface texture descriptors are within normal printed-note ranges "
            f"(GLCM homogeneity={texture_result.global_glcm_homogeneity:.2f}, "
            f"LBP entropy={texture_result.global_lbp_entropy:.2f})."
        )

    return DamageIndicatorResult(
        category_key="texture_irregularity",
        category_name="Surface Texture Irregularity",
        overlay_color_key="texture",
        is_assessed=is_assessed,
        is_detected=is_det,
        severity_score=score if is_assessed else 0.0,
        severity_label=sev_label,
        confidence_label=conf,
        used_reference=texture_result.used_reference,
        mask=mask,
        bounding_boxes=bboxes,
        affected_locations=locs,
        region_count=len(bboxes),
        affected_area_pct=area_pct,
        explanation=expl,
    )


def detect_stain_regions(
    color_result: ColorAnalysisResult,
    region_map: NoteRegionMap,
    quality_report: CaptureQualityReport,
    is_grayscale_input: bool = False,
) -> DamageIndicatorResult:
    """E. Identify localized stain-like regions and chromatic blotches."""
    score = color_result.stain_anomaly_score
    mask = color_result.stain_mask.copy()

    if quality_report.is_overexposed and quality_report.overexposed_fraction > 0.55:
        is_assessed = False
        conf = "Not reliably assessed"
    elif is_grayscale_input or quality_report.is_underexposed or quality_report.is_overexposed:
        is_assessed = True
        conf = "Low"
    elif color_result.used_reference:
        is_assessed = True
        conf = "High"
    else:
        is_assessed = True
        conf = "Moderate"

    is_det, sev_label = _severity_label_from_score(score, is_assessed, threshold=6.5)
    if not is_det:
        mask = np.zeros_like(mask)

    bboxes, locs, area_pct = _extract_bboxes_and_locations(mask, region_map, min_area_px=20)

    if not is_assessed:
        expl = "Not reliably assessed due to extreme exposure clipping."
    elif is_det:
        expl = (
            f"Detected {len(bboxes)} localized stain-like or color-anomaly region(s) "
            f"covering {area_pct:.2f}% of the note surface."
        )
    else:
        expl = "No localized stain-like blotches or abnormal chromatic outliers detected."

    return DamageIndicatorResult(
        category_key="stain_regions",
        category_name="Stain-like Regions & Color Anomalies",
        overlay_color_key="stain_color",
        is_assessed=is_assessed,
        is_detected=is_det,
        severity_score=score if is_assessed else 0.0,
        severity_label=sev_label,
        confidence_label=conf,
        used_reference=color_result.used_reference,
        mask=mask,
        bounding_boxes=bboxes,
        affected_locations=locs,
        region_count=len(bboxes),
        affected_area_pct=area_pct,
        explanation=expl,
    )


def detect_fading_discoloration(
    color_result: ColorAnalysisResult,
    region_map: NoteRegionMap,
    quality_report: CaptureQualityReport,
    is_grayscale_input: bool = False,
) -> DamageIndicatorResult:
    """F. Identify possible fading, washout, or broad yellowing/discoloration."""
    score = color_result.fading_discoloration_score
    mask = color_result.discoloration_mask.copy()

    # Without a reference image, broad fading/discoloration is inherently confounded
    # by camera white balance and ambient lighting if exposure is abnormal
    if (quality_report.is_overexposed or quality_report.is_underexposed) and not color_result.used_reference:
        is_assessed = False
        conf = "Not reliably assessed"
    elif color_result.used_reference and not is_grayscale_input:
        is_assessed = True
        conf = "High"
    elif not color_result.used_reference and not is_grayscale_input:
        is_assessed = True
        conf = "Low"
    else:
        is_assessed = True
        conf = "Low"

    is_det, sev_label = _severity_label_from_score(score, is_assessed, threshold=12.0)
    if not is_det:
        mask = np.zeros_like(mask)

    bboxes, locs, area_pct = _extract_bboxes_and_locations(mask, region_map, min_area_px=30)

    if not is_assessed:
        expl = (
            "Not reliably assessed: camera exposure clipping without an aligned reference "
            "note prevents separating physical ink fading from lighting conditions."
        )
    elif is_det:
        mode_note = (
            "compared against aligned reference note"
            if color_result.used_reference
            else "estimated from low interior contrast/saturation (single-image indicator)"
        )
        expl = f"Possible ink fading or surface discoloration detected ({mode_note})."
    else:
        expl = "Luminance contrast and chromatic saturation show no strong indicators of fading."

    return DamageIndicatorResult(
        category_key="fading_discoloration",
        category_name="Possible Fading or Discoloration",
        overlay_color_key="stain_color",
        is_assessed=is_assessed,
        is_detected=is_det,
        severity_score=score if is_assessed else 0.0,
        severity_label=sev_label,
        confidence_label=conf,
        used_reference=color_result.used_reference,
        mask=mask,
        bounding_boxes=bboxes,
        affected_locations=locs,
        region_count=len(bboxes),
        affected_area_pct=area_pct,
        explanation=expl,
    )


def run_all_damage_detectors(
    edge_result: EdgeAnalysisResult,
    morph_result: MorphologyAnalysisResult,
    texture_result: TextureAnalysisResult,
    color_result: ColorAnalysisResult,
    region_map: NoteRegionMap,
    quality_report: CaptureQualityReport,
    segmentation_succeeded: bool = True,
    is_grayscale_input: bool = False,
) -> Dict[str, DamageIndicatorResult]:
    """Execute all 6 damage-indicator detectors and return a keyed dictionary."""
    return {
        "edge_wear": detect_edge_wear(
            edge_result, region_map, quality_report, segmentation_succeeded
        ),
        "corner_damage": detect_corner_damage(
            edge_result, region_map, quality_report, segmentation_succeeded
        ),
        "fold_lines": detect_fold_lines(morph_result, region_map, quality_report),
        "texture_irregularity": detect_texture_irregularity(
            texture_result, region_map, quality_report
        ),
        "stain_regions": detect_stain_regions(
            color_result, region_map, quality_report, is_grayscale_input
        ),
        "fading_discoloration": detect_fading_discoloration(
            color_result, region_map, quality_report, is_grayscale_input
        ),
    }
