"""Module 8: Color, Stain, and Fading/Discoloration Analysis.

Analyzes chromatic and luminance distributions in HSV and CIE Lab color spaces:
- Illumination normalization before chromatic anomaly evaluation.
- Localized stain-like anomaly detection (comparing local patches and pixels
  against symmetric/neighborhood color statistics so natural printed artwork is
  not misclassified as a stain).
- Global and regional contrast/saturation analysis for potential fading or
  discoloration, enhanced when an aligned reference note is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.preprocessing import normalize_illumination
from src.region_analysis import NoteRegionMap, partition_note_regions


@dataclass
class ColorAnalysisResult:
    """Structured result of HSV/Lab color, stain, and discoloration analysis."""

    mean_hsv_saturation: float
    mean_hsv_value: float
    mean_lab_l: float
    mean_lab_a: float
    mean_lab_b: float
    chroma_spread: float
    stain_anomaly_score: float         # [0, 100], 0 = no stain, 100 = severe staining
    fading_discoloration_score: float  # [0, 100], 0 = normal, 100 = severe fading/yellowing
    color_condition_score: float       # [0, 100], 100 = best condition
    stain_mask: np.ndarray             # Binary uint8 mask of localized stain-like regions
    discoloration_mask: np.ndarray     # Binary uint8 mask of faded/discolored regions
    color_anomaly_heatmap: np.ndarray  # Continuous uint8 anomaly map [0..255]
    used_reference: bool = False
    fading_assessable_confidence: str = "Moderate"  # "High", "Moderate", "Low"
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, float]:
        """Return key color and stain metrics."""
        return {
            "mean_hsv_saturation": round(self.mean_hsv_saturation, 2),
            "mean_hsv_value": round(self.mean_hsv_value, 2),
            "mean_lab_l": round(self.mean_lab_l, 2),
            "mean_lab_a": round(self.mean_lab_a, 2),
            "mean_lab_b": round(self.mean_lab_b, 2),
            "chroma_spread": round(self.chroma_spread, 2),
            "stain_anomaly_score": round(self.stain_anomaly_score, 2),
            "fading_discoloration_score": round(self.fading_discoloration_score, 2),
            "color_condition_score": round(self.color_condition_score, 2),
        }


def analyze_color_and_stains(
    rectified_rgb: np.ndarray,
    rectified_mask: Optional[np.ndarray] = None,
    region_map: Optional[NoteRegionMap] = None,
    reference_rgb: Optional[np.ndarray] = None,
    is_grayscale_input: bool = False,
    sensitivity: float = 1.0,
) -> ColorAnalysisResult:
    """Analyze HSV and CIE Lab spaces for localized stains and fading/discoloration.

    Parameters
    ----------
    rectified_rgb : np.ndarray
        Rectified RGB uint8 note image.
    rectified_mask : np.ndarray, optional
        Foreground binary mask (255 = valid note pixel).
    region_map : NoteRegionMap, optional
        Normalized region partition.
    reference_rgb : np.ndarray, optional
        Aligned reference RGB image of the same dimensions, when available.
    is_grayscale_input : bool
        If True, chromatic channels are degenerate and confidence is reduced.
    sensitivity : float
        Sensitivity multiplier in [0.5, 2.0].
    """
    if rectified_rgb is None or rectified_rgb.size == 0:
        raise ValueError("Cannot run color analysis on an empty image.")

    h, w = rectified_rgb.shape[:2]
    rmap = region_map or partition_note_regions((h, w))
    sens = float(np.clip(sensitivity, 0.5, 2.0))

    # Apply illumination normalization before color comparisons
    norm_rgb = normalize_illumination(rectified_rgb)
    hsv = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(norm_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    valid_bool = (
        (rectified_mask > 0)
        if (rectified_mask is not None and rectified_mask.shape[:2] == (h, w))
        else np.ones((h, w), dtype=bool)
    )
    # Exclude thin outer 4% border from stain detection so background edge bleeding is ignored
    interior_bool = np.zeros((h, w), dtype=bool)
    bh, bw = max(4, int(round(h * 0.04))), max(4, int(round(w * 0.04)))
    interior_bool[bh : h - bh, bw : w - bw] = True
    eval_bool = valid_bool & interior_bool
    if not np.any(eval_bool):
        eval_bool = valid_bool

    s_chan = hsv[:, :, 1]
    v_chan = hsv[:, :, 2]
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1] - 128.0
    b_chan = lab[:, :, 2] - 128.0
    chroma = np.hypot(a_chan, b_chan)

    mean_sat = float(np.mean(s_chan[eval_bool]))
    mean_val = float(np.mean(v_chan[eval_bool]))
    mean_l = float(np.mean(l_chan[eval_bool]))
    mean_a = float(np.mean(lab[:, :, 1][eval_bool]))
    mean_b = float(np.mean(lab[:, :, 2][eval_bool]))
    chroma_spread = float(np.std(chroma[eval_bool]))

    used_ref = (
        reference_rgb is not None
        and isinstance(reference_rgb, np.ndarray)
        and reference_rgb.shape[:2] == (h, w)
    )

    warnings: List[str] = []
    if is_grayscale_input:
        warnings.append(
            "Input image is grayscale; chromatic stain and discoloration detection is "
            "restricted to luminance-only anomalies."
        )

    if used_ref and reference_rgb is not None:
        # Reference-assisted CIE Lab Delta-E comparison
        ref_norm_rgb = normalize_illumination(reference_rgb)
        ref_lab = cv2.cvtColor(ref_norm_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_hsv = cv2.cvtColor(ref_norm_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Smooth slightly to suppress 1-2px registration jitter around fine text lines
        k_blur = max(5, int(round(min(h, w) * 0.02)))
        if k_blur % 2 == 0:
            k_blur += 1
        test_lab_s = cv2.GaussianBlur(lab, (k_blur, k_blur), 0)
        ref_lab_s = cv2.GaussianBlur(ref_lab, (k_blur, k_blur), 0)

        delta_l = test_lab_s[:, :, 0] - ref_lab_s[:, :, 0]
        delta_a = test_lab_s[:, :, 1] - ref_lab_s[:, :, 1]
        delta_b = test_lab_s[:, :, 2] - ref_lab_s[:, :, 2]
        delta_e = np.sqrt(delta_l ** 2 + delta_a ** 2 + delta_b ** 2)

        # Localized stain anomaly: strong local darkening (delta_l < -18) or strong chromatic shift
        stain_field = np.clip((delta_e - 14.0) / 38.0, 0.0, 1.0) * sens
        stain_field = np.where(eval_bool, stain_field, 0.0)

        # Fading / discoloration: compare global/regional contrast & saturation vs reference
        ref_l_std = float(np.std(ref_lab[:, :, 0][eval_bool]))
        test_l_std = float(np.std(l_chan[eval_bool]))
        contrast_loss = max(0.0, (ref_l_std - test_l_std) / max(ref_l_std, 10.0))

        ref_sat_mean = float(np.mean(ref_hsv[:, :, 1][eval_bool]))
        sat_loss = max(0.0, (ref_sat_mean - mean_sat) / max(ref_sat_mean, 15.0))
        yellowingshift = max(0.0, float(np.mean(delta_b[eval_bool])) - 6.0) / 25.0

        fading_score = float(
            np.clip(
                (0.45 * contrast_loss + 0.35 * sat_loss + 0.20 * yellowingshift) * 100.0 * sens,
                0.0,
                100.0,
            )
        )
        discolor_field = np.clip(
            (np.abs(delta_b) - 10.0) / 30.0 + np.maximum(0.0, (ref_lab_s[:, :, 0] - test_lab_s[:, :, 0] - 15.0) / 40.0),
            0.0,
            1.0,
        )
        discolor_field = np.where(eval_bool, discolor_field, 0.0)
        fading_conf = "High" if not is_grayscale_input else "Low"
    else:
        # Single-image mode:
        # Currency notes have printed text and security threads, which we must NOT
        # misclassify as stains. We smooth with a medium kernel (averaging thin text
        # strokes) and measure:
        # 1) Chromatic deviation in CIE Lab (a*, b*) from the note's dominant palette.
        # 2) Localized luminance drop relative to the note's interior paper baseline.
        k_med = max(11, int(round(min(h, w) * 0.04)))
        if k_med % 2 == 0:
            k_med += 1
        k_large = max(61, int(round(min(h, w) * 0.35)))
        if k_large % 2 == 0:
            k_large += 1

        # Use raw rectified_rgb Lab for chromatic comparison so stain contrast is preserved
        raw_lab = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_med = cv2.GaussianBlur(raw_lab, (k_med, k_med), 0)
        lab_large = cv2.GaussianBlur(lab, (k_large, k_large), 0)

        med_a_val = float(np.median(lab_med[:, :, 1][eval_bool]))
        med_b_val = float(np.median(lab_med[:, :, 2][eval_bool]))

        # Chromatic distance from the note's dominant (a*, b*) color center
        ab_dist = np.hypot(
            lab_med[:, :, 1] - med_a_val,
            lab_med[:, :, 2] - med_b_val,
        )
        p90_ab = float(np.percentile(ab_dist[eval_bool], 90.0))
        chroma_thresh = max(10.5, p90_ab * 1.35)
        chroma_outlier = np.clip((ab_dist - chroma_thresh) / 16.0, 0.0, 1.0)

        # Local darkening relative to large-scale background envelope
        local_dark_drop = np.maximum(0.0, lab_large[:, :, 0] - lab_med[:, :, 0])
        dark_smudge = np.clip((local_dark_drop - 24.0) / 30.0, 0.0, 1.0)

        stain_field = np.maximum(dark_smudge * 0.75, chroma_outlier) * sens
        stain_field = np.where(eval_bool, np.clip(stain_field, 0.0, 1.0), 0.0)

        # Estimate possible fading / discoloration in single-image mode:
        l_std = float(np.std(l_chan[eval_bool]))
        low_contrast_indicator = max(0.0, (24.0 - l_std) / 18.0)
        high_washout_indicator = max(0.0, (mean_l - 210.0) / 35.0)
        fading_score = float(
            np.clip(
                (0.65 * low_contrast_indicator + 0.35 * high_washout_indicator) * 65.0 * sens,
                0.0,
                100.0,
            )
        )
        discolor_field = np.zeros((h, w), dtype=np.float32)
        if fading_score >= 15.0:
            l_blur = cv2.GaussianBlur(l_chan, (k_med, k_med), 0)
            local_var = cv2.GaussianBlur((l_chan - l_blur) ** 2, (k_large, k_large), 0)
            local_std_map = np.sqrt(np.maximum(0.0, local_var))
            discolor_field = np.clip((16.0 - local_std_map) / 12.0, 0.0, 1.0)
            discolor_field = np.where(eval_bool, discolor_field, 0.0)

        fading_conf = "Low" if is_grayscale_input else "Moderate"

    # Build uint8 heatmap and binary masks
    color_anomaly_heatmap = np.clip(stain_field * 255.0, 0, 255).astype(np.uint8)
    _, stain_mask = cv2.threshold(color_anomaly_heatmap, 95, 255, cv2.THRESH_BINARY)
    morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, morph_k)
    stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_CLOSE, morph_k)

    # Filter out tiny specks AND narrow vertical security-thread stripes spanning note height
    n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(stain_mask)
    min_stain_area = max(25, int(round(h * w * 0.0005)))
    filtered_stain_mask = np.zeros((h, w), dtype=np.uint8)
    for idx in range(1, n_lbl):
        comp_area = int(stats[idx, cv2.CC_STAT_AREA])
        comp_w = int(stats[idx, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        # Reject vertical security threads (tall and narrow)
        if comp_h > int(h * 0.65) and comp_w < int(w * 0.05):
            continue
        if comp_area >= min_stain_area:
            filtered_stain_mask[lbls == idx] = 255
    stain_mask = filtered_stain_mask

    discolor_u8 = np.clip(discolor_field * 255.0, 0, 255).astype(np.uint8)
    _, discoloration_mask = cv2.threshold(discolor_u8, 120, 255, cv2.THRESH_BINARY)
    discoloration_mask = cv2.morphologyEx(discoloration_mask, cv2.MORPH_OPEN, morph_k)

    # Compute overall stain anomaly score [0, 100] from stain mask coverage and peak intensity
    stain_area_frac = float(np.mean(stain_mask[eval_bool] > 0)) if np.any(eval_bool) else 0.0
    if np.any(stain_mask > 0):
        mean_stain_strength = float(np.mean(stain_field[stain_mask > 0]))
    else:
        mean_stain_strength = 0.0

    raw_stain_score = (
        min(75.0, stain_area_frac * 550.0) + min(25.0, mean_stain_strength * 25.0)
    )
    if stain_area_frac == 0.0:
        raw_stain_score = 0.0
    stain_score = float(np.clip(raw_stain_score, 0.0, 100.0))

    # Combined color/stain condition score [0, 100]
    combined_color_penalty = 0.72 * stain_score + 0.28 * fading_score
    color_condition = float(np.clip(100.0 - combined_color_penalty, 0.0, 100.0))

    explanation = (
        f"Illumination-normalized HSV/Lab analysis yielded color & stain condition score "
        f"{color_condition:.1f}/100 (stain anomaly={stain_score:.1f}%, "
        f"fading/discoloration={fading_score:.1f}%, reference used={used_ref})."
    )

    return ColorAnalysisResult(
        mean_hsv_saturation=mean_sat,
        mean_hsv_value=mean_val,
        mean_lab_l=mean_l,
        mean_lab_a=mean_a,
        mean_lab_b=mean_b,
        chroma_spread=chroma_spread,
        stain_anomaly_score=stain_score,
        fading_discoloration_score=fading_score,
        color_condition_score=color_condition,
        stain_mask=stain_mask,
        discoloration_mask=discoloration_mask,
        color_anomaly_heatmap=color_anomaly_heatmap,
        used_reference=used_ref,
        fading_assessable_confidence=fading_conf,
        explanation=explanation,
        warnings=warnings,
    )
