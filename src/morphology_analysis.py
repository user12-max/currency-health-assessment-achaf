"""Module 7: Morphological Analysis.

Implements scale-aware morphological operations:
- Erosion and Dilation
- Opening and Closing
- Morphological Gradient
- White Top-Hat and Black-Hat transformations
- Multi-orientation linear structuring elements for fold/crease line enhancement
- Structural gap and interior pinhole detection

Structuring element sizes scale proportionally with image dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class DetectedLineSegment:
    """Properties of a candidate fold-like or crease-like linear structure."""

    x1: int
    y1: int
    x2: int
    y2: int
    length_px: float
    normalized_length: float  # Length relative to note height
    orientation_deg: float
    mean_response: float


@dataclass
class MorphologyAnalysisResult:
    """Structured result of scale-aware morphological analysis."""

    kernel_base_size: int
    eroded_gray: np.ndarray
    dilated_gray: np.ndarray
    opened_gray: np.ndarray
    closed_gray: np.ndarray
    morph_gradient: np.ndarray
    tophat_map: np.ndarray
    blackhat_map: np.ndarray
    directional_fold_response: np.ndarray
    fold_candidate_mask: np.ndarray     # Binary uint8 mask of fold-like line structures
    structural_hole_mask: np.ndarray    # Binary uint8 mask of interior holes / gaps
    detected_fold_lines: List[DetectedLineSegment]
    fold_severity_score: float          # [0, 100], 0 = no folds, 100 = severe creases
    fold_condition_score: float         # [0, 100], 100 = best condition
    hole_count: int
    used_reference: bool = False
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)


def _build_directional_line_kernels(length: int) -> List[np.ndarray]:
    """Create 8 directional linear structuring elements spanning 0 to 157.5 degrees."""
    k = max(7, int(length))
    if k % 2 == 0:
        k += 1
    center = k // 2
    radius = center
    kernels: List[np.ndarray] = []
    for angle_deg in (0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5):
        kern = np.zeros((k, k), dtype=np.uint8)
        rad = np.radians(angle_deg)
        dx = int(round(radius * np.cos(rad)))
        dy = int(round(radius * np.sin(rad)))
        cv2.line(
            kern,
            (center - dx, center - dy),
            (center + dx, center + dy),
            1,
            thickness=2,
        )
        kernels.append(kern)
    return kernels


def _compute_anisotropic_fold_response(
    gray: np.ndarray,
    base_ksize: int,
) -> np.ndarray:
    """Highlight continuous physical crease troughs and ridges while rejecting text and ink blocks.

    Key morphological insight:
    1. Disconnected printed text characters have gaps between letters where `blackhat == 0`.
       Applying directional line opening (`MORPH_OPEN` with a long 1D line kernel `dk`)
       directly to the trough map eliminates broken character strings while preserving
       continuous fold lines.
    2. Flat printed security threads or border stripes are purely absorptive (never
       brighter than the local paper background), whereas physical creases exhibit an
       adjacent bright highlight/wear ridge above the local paper median.
    """
    iso_k = max(5, base_ksize)
    if iso_k % 2 == 0:
        iso_k += 1
    ell_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (iso_k, iso_k))

    gray_f = gray.astype(np.float32)
    # Estimate local paper baseline intensity using a wider median/Gaussian blur
    bg_k = max(15, iso_k * 3)
    if bg_k % 2 == 0:
        bg_k += 1
    paper_base = cv2.GaussianBlur(gray_f, (bg_k, bg_k), 0)

    # True dark trough below paper baseline & true bright ridge above paper baseline
    trough_map = np.maximum(0.0, paper_base - gray_f)
    ridge_map = np.maximum(0.0, gray_f - paper_base)

    # Suppress wide solid ink blocks by opening with an isotropic kernel
    trough_thin = np.maximum(
        0.0, trough_map - cv2.morphologyEx(trough_map, cv2.MORPH_OPEN, ell_kernel)
    )
    ridge_thin = np.maximum(
        0.0, ridge_map - cv2.morphologyEx(ridge_map, cv2.MORPH_OPEN, ell_kernel)
    )

    line_len = max(15, iso_k * 3 + 1)
    dir_kernels = _build_directional_line_kernels(line_len)
    couple_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (iso_k, iso_k))

    dir_responses = []
    for dk in dir_kernels:
        # Directional opening on trough_thin and ridge_thin separately so text gaps break
        cont_trough = cv2.morphologyEx(trough_thin, cv2.MORPH_OPEN, dk)
        cont_ridge = cv2.morphologyEx(ridge_thin, cv2.MORPH_OPEN, dk)

        # Require coupled continuous trough + adjacent continuous ridge
        ridge_dil = cv2.dilate(cont_ridge, couple_k)
        coupled_dir = np.sqrt(cont_trough * np.maximum(ridge_dil, 1.0)) * (
            np.clip(ridge_dil / 4.0, 0.15, 1.5)
        )
        dir_responses.append(coupled_dir)

    stack_dir = np.stack(dir_responses, axis=0)
    max_dir = np.max(stack_dir, axis=0)
    min_dir = np.min(stack_dir, axis=0)
    anisotropic = np.maximum(0.0, max_dir - 0.75 * min_dir) * 3.2

    return np.clip(anisotropic, 0, 255).astype(np.uint8)


def _deduplicate_line_segments(
    segments: List[DetectedLineSegment],
    dist_thresh: float = 18.0,
    angle_thresh: float = 12.0,
) -> List[DetectedLineSegment]:
    """Merge/suppress nearly collinear, overlapping Hough line segments."""
    if not segments:
        return []
    # Sort by length * response descending
    sorted_segs = sorted(
        segments, key=lambda s: s.length_px * s.mean_response, reverse=True
    )
    kept: List[DetectedLineSegment] = []
    for cand in sorted_segs:
        cx = 0.5 * (cand.x1 + cand.x2)
        cy = 0.5 * (cand.y1 + cand.y2)
        is_dup = False
        for existing in kept:
            ex = 0.5 * (existing.x1 + existing.x2)
            ey = 0.5 * (existing.y1 + existing.y2)
            angle_diff = abs(cand.orientation_deg - existing.orientation_deg)
            angle_diff = min(angle_diff, 180.0 - angle_diff)
            center_dist = float(np.hypot(cx - ex, cy - ey))
            if angle_diff <= angle_thresh and center_dist <= dist_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    return kept


def analyze_morphology(
    rectified_rgb: np.ndarray,
    rectified_mask: Optional[np.ndarray] = None,
    reference_gray: Optional[np.ndarray] = None,
    scale_factor: float = 1.0,
    fold_sensitivity: float = 1.0,
) -> MorphologyAnalysisResult:
    """Perform scale-aware morphological operations and fold/crease line analysis.

    Parameters
    ----------
    rectified_rgb : np.ndarray
        Rectified RGB uint8 note image.
    rectified_mask : np.ndarray, optional
        Foreground binary mask (255 = note foreground).
    reference_gray : np.ndarray, optional
        Aligned reference grayscale image when available.
    scale_factor : float
        Multiplier for structuring element scale.
    fold_sensitivity : float
        Sensitivity multiplier for fold/crease detection in [0.5, 2.0].
    """
    if rectified_rgb is None or rectified_rgb.size == 0:
        raise ValueError("Cannot run morphological analysis on an empty image.")

    gray = (
        rectified_rgb
        if rectified_rgb.ndim == 2
        else cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2GRAY)
    )
    h, w = gray.shape[:2]
    sf = float(np.clip(scale_factor, 0.5, 2.5))
    sens = float(np.clip(fold_sensitivity, 0.5, 2.0))

    # Scale-aware structuring element size (~1.5% of shortest note dimension)
    base_k = max(3, int(round(min(h, w) * 0.015 * sf)))
    if base_k % 2 == 0:
        base_k += 1

    se_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (base_k, base_k))
    se_ell = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (base_k * 2 + 1, base_k * 2 + 1)
    )

    # 1. Core morphological operations
    eroded = cv2.erode(gray, se_rect)
    dilated = cv2.dilate(gray, se_rect)
    opened = cv2.morphologyEx(gray, se_rect, cv2.MORPH_OPEN) if False else cv2.morphologyEx(gray, cv2.MORPH_OPEN, se_rect)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, se_rect)
    morph_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, se_rect)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_ell)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se_ell)

    # 2. Directional fold/crease response
    fold_resp = _compute_anisotropic_fold_response(gray, base_k * 2 + 1)

    used_ref = (
        reference_gray is not None
        and isinstance(reference_gray, np.ndarray)
        and reference_gray.shape[:2] == (h, w)
    )
    if used_ref and reference_gray is not None:
        ref_fold_resp = _compute_anisotropic_fold_response(reference_gray, base_k * 2 + 1)
        # Suppress printed linear artwork present in the clean reference note
        diff_resp = np.maximum(
            0.0, fold_resp.astype(np.float32) - ref_fold_resp.astype(np.float32) * 1.15
        )
        fold_resp = np.clip(diff_resp, 0, 255).astype(np.uint8)

    # Exclude outer 8% boundary so outer frame borders are not detected as internal folds
    interior_mask = np.zeros((h, w), dtype=np.uint8)
    mh = max(8, int(round(h * 0.08)))
    mw = max(8, int(round(w * 0.08)))
    interior_mask[mh : h - mh, mw : w - mw] = 255
    if rectified_mask is not None:
        erode_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mh, mh))
        eroded_fg = cv2.erode(rectified_mask, erode_b)
        interior_mask = cv2.bitwise_and(interior_mask, eroded_fg)

    masked_fold_resp = cv2.bitwise_and(fold_resp, interior_mask)

    # 3. Extract elongated line segments via adaptive threshold + HoughLinesP
    valid_vals = masked_fold_resp[interior_mask > 0]
    if valid_vals.size > 0:
        p98 = float(np.percentile(valid_vals, 98.0))
    else:
        p98 = 20.0

    # Require a meaningful absolute coupled contrast so clean notes aren't triggered
    min_abs_thresh = max(16.0, 24.0 / sens)
    thresh_val = max(min_abs_thresh, p98 * (0.90 / (sens ** 0.35)))
    _, raw_cand = cv2.threshold(
        masked_fold_resp, int(round(min(245.0, thresh_val))), 255, cv2.THRESH_BINARY
    )

    # Probabilistic Hough Transform on candidate pixels to find long linear creases
    min_line_len = max(int(round(h * 0.32)), 55)
    max_line_gap = max(int(round(h * 0.04)), 8)
    lines = cv2.HoughLinesP(
        raw_cand,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=max(28, int(round(min_line_len * 0.45))),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    fold_candidate_mask = np.zeros((h, w), dtype=np.uint8)
    raw_segments: List[DetectedLineSegment] = []

    if lines is not None:
        for line_arr in lines:
            x1, y1, x2, y2 = [int(v) for v in np.asarray(line_arr).ravel()[:4]]
            length_px = float(np.hypot(x2 - x1, y2 - y1))
            if length_px < min_line_len:
                continue

            line_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=max(3, base_k))
            mean_resp = float(np.mean(masked_fold_resp[line_mask > 0]))
            if mean_resp < min_abs_thresh * 0.90:
                continue

            angle_deg = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
            norm_len = length_px / float(max(1, h))

            raw_segments.append(
                DetectedLineSegment(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    length_px=length_px,
                    normalized_length=norm_len,
                    orientation_deg=angle_deg,
                    mean_response=mean_resp,
                )
            )

    detected_segments = _deduplicate_line_segments(raw_segments)
    for seg in detected_segments:
        cv2.line(
            fold_candidate_mask,
            (seg.x1, seg.y1),
            (seg.x2, seg.y2),
            255,
            thickness=max(4, base_k + 1),
        )

    # 4. Detect interior structural holes / voids (if any exist inside the note contour)
    structural_hole_mask = np.zeros((h, w), dtype=np.uint8)
    hole_count = 0
    if rectified_mask is not None:
        cnts, _ = cv2.findContours(rectified_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            filled_outer = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(filled_outer, cnts, -1, 255, thickness=cv2.FILLED)
            holes = cv2.bitwise_and(filled_outer, cv2.bitwise_not(rectified_mask))
            holes = cv2.bitwise_and(holes, interior_mask)
            n_h, h_lbls, h_stats, _ = cv2.connectedComponentsWithStats(holes)
            for idx in range(1, n_h):
                if h_stats[idx, cv2.CC_STAT_AREA] >= 12:
                    structural_hole_mask[h_lbls == idx] = 255
                    hole_count += 1

    # 5. Compute fold/crease severity score in [0, 100]
    mask_area_ratio = float(np.mean(fold_candidate_mask > 0))
    total_norm_len = sum(seg.normalized_length for seg in detected_segments)
    max_seg_resp = max((seg.mean_response for seg in detected_segments), default=0.0)

    if len(detected_segments) == 0 and hole_count == 0:
        fold_severity = 0.0
    else:
        raw_severity = (
            min(55.0, total_norm_len * 22.0)
            + min(30.0, mask_area_ratio * 850.0)
            + min(15.0, (max_seg_resp / 48.0) * 15.0)
        ) * sens
        if hole_count > 0:
            raw_severity += min(25.0, hole_count * 12.0)
        fold_severity = float(np.clip(raw_severity, 0.0, 100.0))

    fold_condition = float(np.clip(100.0 - fold_severity, 0.0, 100.0))

    explanation = (
        f"Morphological top-hat/black-hat and directional line analysis yielded fold/crease "
        f"condition score {fold_condition:.1f}/100 ({len(detected_segments)} elongated line "
        f"segment(s) detected, interior holes={hole_count})."
    )

    return MorphologyAnalysisResult(
        kernel_base_size=base_k,
        eroded_gray=eroded,
        dilated_gray=dilated,
        opened_gray=opened,
        closed_gray=closed,
        morph_gradient=morph_gradient,
        tophat_map=tophat,
        blackhat_map=blackhat,
        directional_fold_response=masked_fold_resp,
        fold_candidate_mask=fold_candidate_mask,
        structural_hole_mask=structural_hole_mask,
        detected_fold_lines=detected_segments,
        fold_severity_score=fold_severity,
        fold_condition_score=fold_condition,
        hole_count=hole_count,
        used_reference=used_ref,
        explanation=explanation,
    )
