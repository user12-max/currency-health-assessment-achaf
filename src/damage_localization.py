"""Module 18: Damage Localization and Multi-Category Spatial Overlay.

Processes binary indicator masks to:
1. Remove isolated noise and apply morphological cleanup.
2. Extract connected components, contours, bounding boxes, and region properties.
3. Generate a combined multi-category damage map and annotated note overlay in
   rectified note coordinates.
4. Map localized masks and annotations back to the original (resized) image
   coordinate system via inverse homography so perspective transformation never
   causes coordinate misalignment.

Color Legend (RGB):
- Red (220, 38, 38): Edge or corner irregularity
- Blue (37, 99, 235): Fold-like line indicators
- Yellow (234, 179, 8): Stain-like or color anomaly indicators
- Purple (147, 51, 234): Surface texture irregularity indicators
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import DAMAGE_CATEGORY_LABELS, DAMAGE_COLOR_LEGEND_RGB
from src.damage_detection import DamageIndicatorResult
from src.region_analysis import NoteRegionMap, partition_note_regions


@dataclass
class LocalizedDefectRegion:
    """Properties of a single localized connected damage region."""

    region_id: int
    category_key: str
    category_name: str
    color_key: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in rectified coordinates
    centroid: Tuple[float, float]
    area_px: int
    area_pct: float
    anatomical_location: str
    confidence_label: str


@dataclass
class DamageLocalizationResult:
    """Complete spatial localization outputs, masks, and visual overlays."""

    annotated_rectified_rgb: np.ndarray
    annotated_original_rgb: np.ndarray
    combined_damage_map_rgb: np.ndarray
    category_masks: Dict[str, np.ndarray]  # Keys: "edge_corner", "fold_line", "stain_color", "texture"
    localized_regions: List[LocalizedDefectRegion] = field(default_factory=list)
    total_damaged_area_pct: float = 0.0
    has_detected_damage: bool = False
    legend_rgb: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: dict(DAMAGE_COLOR_LEGEND_RGB)
    )
    legend_labels: Dict[str, str] = field(
        default_factory=lambda: dict(DAMAGE_CATEGORY_LABELS)
    )


def _clean_binary_mask(mask: np.ndarray, min_area_px: int = 12) -> np.ndarray:
    """Apply morphological cleanup and remove small isolated noise components."""
    if mask is None or mask.size == 0 or not np.any(mask > 0):
        return np.zeros_like(mask, dtype=np.uint8) if mask is not None else np.zeros((1, 1), dtype=np.uint8)

    binary = (mask > 0).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    cleaned = np.zeros_like(binary)
    for lbl in range(1, num_labels):
        if int(stats[lbl, cv2.CC_STAT_AREA]) >= min_area_px:
            cleaned[labels == lbl] = 255
    return cleaned


def localize_damage_regions(
    rectified_rgb: np.ndarray,
    original_resized_rgb: np.ndarray,
    indicators: Dict[str, DamageIndicatorResult],
    region_map: Optional[NoteRegionMap] = None,
    inv_homography: Optional[np.ndarray] = None,
) -> DamageLocalizationResult:
    """Generate cleaned category masks, region properties, and overlays in both coordinate spaces.

    Parameters
    ----------
    rectified_rgb : np.ndarray
        Rectified currency note image (H_r, W_r, 3) in RGB.
    original_resized_rgb : np.ndarray
        Preprocessed input image before perspective warp (H_o, W_o, 3) in RGB.
    indicators : dict of str -> DamageIndicatorResult
        Output from `run_all_damage_detectors`.
    region_map : NoteRegionMap, optional
        Normalized region partition of the rectified note.
    inv_homography : np.ndarray, optional
        3x3 inverse perspective matrix mapping rectified -> original_resized coordinates.
    """
    if rectified_rgb is None or rectified_rgb.size == 0:
        raise ValueError("rectified_rgb cannot be empty for damage localization.")

    hr, wr = rectified_rgb.shape[:2]
    ho, wo = original_resized_rgb.shape[:2]
    rmap = region_map or partition_note_regions((hr, wr))

    # Initialize the 4 canonical color-legend category masks
    category_masks: Dict[str, np.ndarray] = {
        "edge_corner": np.zeros((hr, wr), dtype=np.uint8),
        "fold_line": np.zeros((hr, wr), dtype=np.uint8),
        "stain_color": np.zeros((hr, wr), dtype=np.uint8),
        "texture": np.zeros((hr, wr), dtype=np.uint8),
    }

    localized_regions: List[LocalizedDefectRegion] = []
    region_counter = 1
    total_area_f = float(max(1, hr * wr))

    for ind_key, ind in indicators.items():
        if not ind.is_assessed or not ind.is_detected or ind.mask is None:
            continue
        if ind.mask.shape[:2] != (hr, wr):
            resized_m = cv2.resize(ind.mask, (wr, hr), interpolation=cv2.INTER_NEAREST)
        else:
            resized_m = ind.mask

        cleaned_m = _clean_binary_mask(resized_m, min_area_px=12)
        if not np.any(cleaned_m > 0):
            continue

        color_key = ind.overlay_color_key
        if color_key in category_masks:
            category_masks[color_key] = cv2.bitwise_or(category_masks[color_key], cleaned_m)

        # Extract connected components for this specific indicator
        n_lbl, lbls, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned_m, connectivity=8
        )
        for lbl in range(1, n_lbl):
            area_px = int(stats[lbl, cv2.CC_STAT_AREA])
            if area_px < 12:
                continue
            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
            bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[lbl][0]), float(centroids[lbl][1])
            loc_str = rmap.locate_point_region(cx, cy)

            localized_regions.append(
                LocalizedDefectRegion(
                    region_id=region_counter,
                    category_key=ind_key,
                    category_name=ind.category_name,
                    color_key=color_key,
                    bbox=(x, y, bw, bh),
                    centroid=(cx, cy),
                    area_px=area_px,
                    area_pct=(float(area_px) / total_area_f) * 100.0,
                    anatomical_location=loc_str,
                    confidence_label=ind.confidence_label,
                )
            )
            region_counter += 1

    # Build combined damage map (dark slate background with color-coded layers)
    # Layer priority from broad surface to sharp structural features:
    # texture (purple) -> stain_color (yellow) -> fold_line (blue) -> edge_corner (red)
    draw_order = ["texture", "stain_color", "fold_line", "edge_corner"]

    # Darkened base of rectified note so context is visible in the damage map
    dark_base = (rectified_rgb.astype(np.float32) * 0.25).astype(np.uint8)
    combined_damage_map = dark_base.copy()
    annotated_rect = rectified_rgb.copy()

    union_mask = np.zeros((hr, wr), dtype=bool)

    for ckey in draw_order:
        cmask = category_masks[ckey]
        if not np.any(cmask > 0):
            continue
        rgb_col = np.array(DAMAGE_COLOR_LEGEND_RGB[ckey], dtype=np.uint8)
        active = cmask > 0
        union_mask |= active

        # Solid vivid color on diagnostic combined damage map
        combined_damage_map[active] = rgb_col

        # Semi-transparent tint (42% color + 58% image) on annotated rectified note
        annotated_rect[active] = np.clip(
            0.58 * annotated_rect[active].astype(np.float32)
            + 0.42 * rgb_col.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)

        # Crisp contour outline around each detected region
        cnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            annotated_rect,
            cnts,
            -1,
            tuple(int(v) for v in rgb_col),
            thickness=2,
        )
        cv2.drawContours(
            combined_damage_map,
            cnts,
            -1,
            (255, 255, 255),
            thickness=1,
        )

    # Draw bounding boxes and compact ID tags for top regions on annotated_rect
    for reg in localized_regions[:20]:
        rx, ry, rbw, rbh = reg.bbox
        col_tuple = tuple(int(v) for v in DAMAGE_COLOR_LEGEND_RGB[reg.color_key])
        cv2.rectangle(
            annotated_rect,
            (rx, ry),
            (rx + rbw, ry + rbh),
            col_tuple,
            thickness=1,
        )
        label_txt = f"#{reg.region_id}"
        ty = max(12, ry - 4)
        cv2.putText(
            annotated_rect,
            label_txt,
            (rx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            col_tuple,
            1,
            cv2.LINE_AA,
        )

    # Map overlays back to original_resized_rgb via inverse homography
    annotated_orig = original_resized_rgb.copy()
    if np.any(union_mask):
        if inv_homography is not None and inv_homography.shape == (3, 3):
            for ckey in draw_order:
                cmask = category_masks[ckey]
                if not np.any(cmask > 0):
                    continue
                warped_back = cv2.warpPerspective(
                    cmask,
                    inv_homography,
                    (wo, ho),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                active_orig = warped_back > 0
                rgb_col = np.array(DAMAGE_COLOR_LEGEND_RGB[ckey], dtype=np.uint8)
                annotated_orig[active_orig] = np.clip(
                    0.58 * annotated_orig[active_orig].astype(np.float32)
                    + 0.42 * rgb_col.astype(np.float32),
                    0,
                    255,
                ).astype(np.uint8)
                cnts_o, _ = cv2.findContours(
                    (active_orig.astype(np.uint8) * 255),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                cv2.drawContours(
                    annotated_orig,
                    cnts_o,
                    -1,
                    tuple(int(v) for v in rgb_col),
                    thickness=2,
                )
        elif (ho, wo) == (hr, wr):
            annotated_orig = annotated_rect.copy()

    total_damaged_pct = float(np.mean(union_mask)) * 100.0
    has_damage = bool(np.any(union_mask))

    return DamageLocalizationResult(
        annotated_rectified_rgb=annotated_rect,
        annotated_original_rgb=annotated_orig,
        combined_damage_map_rgb=combined_damage_map,
        category_masks=category_masks,
        localized_regions=localized_regions,
        total_damaged_area_pct=total_damaged_pct,
        has_detected_damage=has_damage,
    )
