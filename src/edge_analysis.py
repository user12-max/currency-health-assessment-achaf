"""Module 6: Edge and Corner Boundary Analysis.

Implements Canny edge detection, Sobel gradient analysis, contour boundary
inspection, and four-edge / four-corner integrity measurements.

Conceptual separation:
1. External note boundary (evaluated via the rectified foreground mask and outer
   border bands to measure edge wear, tears, bites/nicks, and missing corners).
2. Internal edges (Canny and Sobel gradients inside the note body, separated from
   the outer boundary so printed artwork is never mistaken for a torn outer edge).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.region_analysis import NoteRegionMap, partition_note_regions


@dataclass
class SingleEdgeMetrics:
    """Boundary integrity measurements for one of the four outer edges."""

    edge_name: str  # "top", "bottom", "left", "right"
    continuity_ratio: float       # [0.0, 1.0], 1.0 = unbroken edge
    missing_area_fraction: float  # [0.0, 1.0], fraction of edge band missing/eroded
    roughness_px: float           # Standard deviation of boundary displacement in pixels
    edge_wear_score: float        # [0.0, 100.0], 0 = intact, 100 = severely worn
    edge_condition_score: float   # [0.0, 100.0], 100 = intact


@dataclass
class SingleCornerMetrics:
    """Geometry and area-retention measurements for one of the four note corners."""

    corner_name: str  # "top_left", "top_right", "bottom_left", "bottom_right"
    area_retention_ratio: float   # [0.0, 1.0], 1.0 = full rectangular corner present
    missing_fraction: float       # [0.0, 1.0], 1.0 - area_retention_ratio
    tip_distance_norm: float      # Normalized distance from true corner vertex to note pixel
    corner_damage_score: float    # [0.0, 100.0], 0 = intact, 100 = severely missing/folded
    corner_condition_score: float # [0.0, 100.0], 100 = intact


@dataclass
class EdgeAnalysisResult:
    """Complete result of external boundary, corner, and internal edge analysis."""

    edge_metrics: Dict[str, SingleEdgeMetrics]
    corner_metrics: Dict[str, SingleCornerMetrics]
    overall_edge_wear_score: float        # [0, 100], higher = more wear
    overall_edge_condition_score: float   # [0, 100], higher = healthier edges
    overall_corner_damage_score: float    # [0, 100], higher = more corner damage
    overall_corner_condition_score: float # [0, 100], higher = healthier corners
    internal_edge_density: float
    mean_sobel_gradient: float
    canny_edges: np.ndarray               # uint8 binary map of internal/global Canny edges
    sobel_magnitude_vis: np.ndarray       # uint8 visualization of Sobel gradient magnitude
    edge_wear_mask: np.ndarray            # uint8 binary mask of detected boundary edge defects
    corner_damage_mask: np.ndarray        # uint8 binary mask of detected corner defects
    boundary_visualization_rgb: np.ndarray
    is_boundary_assessable: bool = True
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)


def _refine_note_border_mask(
    rectified_rgb: np.ndarray,
    rectified_mask: np.ndarray,
) -> np.ndarray:
    """Refine the foreground mask along the outer border bands using color/luminance contrast.

    When an image is already cropped or segmented with a bounding box, dark or uniform
    background triangles at torn corners or nicks along the edge may still be included
    inside the initial mask if the outer quad covered them. This helper inspects the
    outer border against the central note distribution to expose missing corner/edge bites.
    """
    h, w = rectified_rgb.shape[:2]
    refined = (rectified_mask > 0).astype(np.uint8) * 255

    lab = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Sample central interior note pixels to characterize note paper/ink distribution
    cy1, cy2 = int(h * 0.20), int(h * 0.80)
    cx1, cx2 = int(w * 0.20), int(w * 0.80)
    center_lab = lab[cy1:cy2, cx1:cx2].reshape(-1, 3)
    center_gray = gray[cy1:cy2, cx1:cx2].ravel()

    p5_gray = float(np.percentile(center_gray, 3.0))
    p98_gray = float(np.percentile(center_gray, 98.0))
    med_lab = np.median(center_lab, axis=0)
    std_lab = np.maximum(np.std(center_lab, axis=0), 8.0)

    # Check outer 12% perimeter for obvious background intrusions (very dark black border
    # from perspective warp or strongly contrasting background at torn corners/edges)
    border_zone = np.zeros((h, w), dtype=bool)
    bh = max(6, int(round(h * 0.14)))
    bw = max(6, int(round(w * 0.14)))
    border_zone[:bh, :] = True
    border_zone[-bh:, :] = True
    border_zone[:, :bw] = True
    border_zone[:, -bw:] = True

    # Pixels that are near-black (e.g., < max(12, p5_gray - 28)) or extreme background
    # outliers connected to the outer image boundary
    dark_bg = gray < max(12.0, p5_gray - 30.0)
    z_dist = np.linalg.norm((lab - med_lab) / std_lab, axis=2)
    # Smooth local variance: background voids at torn corners are typically smooth/uniform
    local_mean = cv2.blur(gray, (7, 7))
    local_sq = cv2.blur(gray * gray, (7, 7))
    local_std = np.sqrt(np.maximum(0.0, local_sq - local_mean * local_mean))

    candidate_bg = border_zone & (
        dark_bg | ((z_dist > 4.2) & (local_std < 6.0))
    )

    # Only keep candidate background components that touch the 2-pixel outer frame
    cand_u8 = candidate_bg.astype(np.uint8) * 255
    num_labels, labels = cv2.connectedComponents(cand_u8)
    touch_border_bg = np.zeros((h, w), dtype=bool)
    for lbl in range(1, num_labels):
        comp = labels == lbl
        if (
            np.any(comp[:2, :])
            or np.any(comp[-2:, :])
            or np.any(comp[:, :2])
            or np.any(comp[:, -2:])
        ):
            touch_border_bg |= comp

    refined[touch_border_bg] = 0
    return refined


def analyze_edges_and_corners(
    rectified_rgb: np.ndarray,
    rectified_mask: np.ndarray,
    region_map: Optional[NoteRegionMap] = None,
    segmentation_succeeded: bool = True,
    sensitivity: float = 1.0,
) -> EdgeAnalysisResult:
    """Analyze outer boundary edge wear, corner integrity, and internal edge gradients.

    Parameters
    ----------
    rectified_rgb : np.ndarray
        Rectified RGB uint8 note image.
    rectified_mask : np.ndarray
        Foreground mask of the rectified note (255 = note, 0 = missing/background).
    region_map : NoteRegionMap, optional
        Normalized region partition.
    segmentation_succeeded : bool
        Whether note segmentation isolated the note boundary.
    sensitivity : float
        Sensitivity multiplier in [0.5, 2.0].
    """
    if rectified_rgb is None or rectified_rgb.size == 0:
        raise ValueError("Cannot analyze edges on an empty image.")

    h, w = rectified_rgb.shape[:2]
    rmap = region_map or partition_note_regions((h, w))
    sens = float(np.clip(sensitivity, 0.5, 2.0))

    gray = cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 1. Canny edge detection and Sobel gradient magnitude
    canny_edges = cv2.Canny(blurred, 45, 135)
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    mean_sobel = float(np.mean(sobel_mag))

    p99 = max(float(np.percentile(sobel_mag, 99.0)), 1.0)
    sobel_vis = np.clip((sobel_mag / p99) * 255.0, 0, 255).astype(np.uint8)

    # Measure internal edge density strictly inside the central region
    central_canny = rmap.central_region.extract_slice(canny_edges)
    internal_edge_density = float(np.mean(central_canny > 0))

    # 2. Refine foreground mask along the perimeter to capture nicks, tears, and missing corners
    refined_mask = _refine_note_border_mask(rectified_rgb, rectified_mask)
    fg_bool = refined_mask > 0

    edge_wear_mask = np.zeros((h, w), dtype=np.uint8)
    corner_damage_mask = np.zeros((h, w), dtype=np.uint8)

    edge_metrics: Dict[str, SingleEdgeMetrics] = {}
    for ename, ebox in rmap.edge_regions.items():
        eband = ebox.extract_slice(fg_bool)
        band_h, band_w = eband.shape[:2]

        # Missing fraction inside the edge band
        missing_frac = float(1.0 - np.mean(eband))

        # Measure boundary displacement profile (distance from outer border to first foreground pixel)
        if ename == "top":
            has_fg = np.any(eband, axis=0)
            first_idx = np.argmax(eband, axis=0).astype(np.float32)
            first_idx[~has_fg] = float(band_h)
            continuity = float(np.mean(first_idx <= max(2.0, band_h * 0.18)))
            roughness = float(np.std(first_idx)) / max(1.0, float(band_h))
        elif ename == "bottom":
            flipped = eband[::-1, :]
            has_fg = np.any(flipped, axis=0)
            first_idx = np.argmax(flipped, axis=0).astype(np.float32)
            first_idx[~has_fg] = float(band_h)
            continuity = float(np.mean(first_idx <= max(2.0, band_h * 0.18)))
            roughness = float(np.std(first_idx)) / max(1.0, float(band_h))
        elif ename == "left":
            has_fg = np.any(eband, axis=1)
            first_idx = np.argmax(eband, axis=1).astype(np.float32)
            first_idx[~has_fg] = float(band_w)
            continuity = float(np.mean(first_idx <= max(2.0, band_w * 0.18)))
            roughness = float(np.std(first_idx)) / max(1.0, float(band_w))
        else:  # "right"
            flipped = eband[:, ::-1]
            has_fg = np.any(flipped, axis=1)
            first_idx = np.argmax(flipped, axis=1).astype(np.float32)
            first_idx[~has_fg] = float(band_w)
            continuity = float(np.mean(first_idx <= max(2.0, band_w * 0.18)))
            roughness = float(np.std(first_idx)) / max(1.0, float(band_w))

        # Also check high-frequency jaggedness / tear notches along the edge strip
        # Ignore small <=1.5% border tolerance from perspective discretization
        eff_missing = max(0.0, missing_frac - 0.015)
        eff_discontinuity = max(0.0, (1.0 - continuity) - 0.03)
        eff_roughness = max(0.0, roughness - 0.04)

        raw_wear = (
            (eff_missing * 260.0) + (eff_discontinuity * 95.0) + (eff_roughness * 140.0)
        ) * sens
        wear_score = float(np.clip(raw_wear, 0.0, 100.0))
        cond_score = float(np.clip(100.0 - wear_score, 0.0, 100.0))

        edge_metrics[ename] = SingleEdgeMetrics(
            edge_name=ename,
            continuity_ratio=continuity,
            missing_area_fraction=missing_frac,
            roughness_px=roughness * float(max(band_h, band_w)),
            edge_wear_score=wear_score,
            edge_condition_score=cond_score,
        )

        # Populate spatial edge_wear_mask where pixels are missing in the edge box
        # beyond a 2-pixel discretization margin
        missing_in_box = (~eband).astype(np.uint8) * 255
        if ename == "top" and band_h > 2:
            missing_in_box[:2, :] = 0
        elif ename == "bottom" and band_h > 2:
            missing_in_box[-2:, :] = 0
        elif ename == "left" and band_w > 2:
            missing_in_box[:, :2] = 0
        elif ename == "right" and band_w > 2:
            missing_in_box[:, -2:] = 0

        if wear_score >= 6.0:
            edge_wear_mask[ebox.y1 : ebox.y2, ebox.x1 : ebox.x2] = np.maximum(
                edge_wear_mask[ebox.y1 : ebox.y2, ebox.x1 : ebox.x2],
                missing_in_box,
            )

    # 3. Four-corner geometry and area-retention analysis
    corner_metrics: Dict[str, SingleCornerMetrics] = {}
    for cname, cbox in rmap.corner_regions.items():
        cband = cbox.extract_slice(fg_bool)
        ch, cw = cband.shape[:2]
        retention = float(np.mean(cband))
        missing_frac = float(1.0 - retention)

        # Compute normalized distance from exact corner vertex to nearest foreground pixel
        ys, xs = np.nonzero(cband)
        if ys.size == 0:
            tip_dist_norm = 1.0
        else:
            vy = 0.0 if "top" in cname else float(ch - 1)
            vx = 0.0 if "left" in cname else float(cw - 1)
            dists = np.sqrt(((ys - vy) / max(1.0, float(ch))) ** 2 + ((xs - vx) / max(1.0, float(cw))) ** 2)
            tip_dist_norm = float(np.min(dists))

        eff_missing = max(0.0, missing_frac - 0.02)
        eff_tip = max(0.0, tip_dist_norm - 0.05)

        raw_corner_dmg = ((eff_missing * 240.0) + (eff_tip * 130.0)) * sens
        dmg_score = float(np.clip(raw_corner_dmg, 0.0, 100.0))
        cond_score = float(np.clip(100.0 - dmg_score, 0.0, 100.0))

        corner_metrics[cname] = SingleCornerMetrics(
            corner_name=cname,
            area_retention_ratio=retention,
            missing_fraction=missing_frac,
            tip_distance_norm=tip_dist_norm,
            corner_damage_score=dmg_score,
            corner_condition_score=cond_score,
        )

        if dmg_score >= 6.0:
            missing_corner = (~cband).astype(np.uint8) * 255
            corner_damage_mask[cbox.y1 : cbox.y2, cbox.x1 : cbox.x2] = np.maximum(
                corner_damage_mask[cbox.y1 : cbox.y2, cbox.x1 : cbox.x2],
                missing_corner,
            )

    # Clean up tiny 1-2 px border artifacts in edge & corner masks
    clean_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_wear_mask = cv2.morphologyEx(edge_wear_mask, cv2.MORPH_OPEN, clean_k)
    corner_damage_mask = cv2.morphologyEx(corner_damage_mask, cv2.MORPH_OPEN, clean_k)

    # Aggregate 4 edges and 4 corners (weighting worst edge/corner + mean so localized severe
    # tears are not diluted across 3 intact edges)
    e_wears = [m.edge_wear_score for m in edge_metrics.values()]
    overall_edge_wear = float(
        np.clip(0.55 * np.mean(e_wears) + 0.45 * np.max(e_wears), 0.0, 100.0)
    )
    overall_edge_cond = float(np.clip(100.0 - overall_edge_wear, 0.0, 100.0))

    c_dmgs = [m.corner_damage_score for m in corner_metrics.values()]
    overall_corner_dmg = float(
        np.clip(0.50 * np.mean(c_dmgs) + 0.50 * np.max(c_dmgs), 0.0, 100.0)
    )
    overall_corner_cond = float(np.clip(100.0 - overall_corner_dmg, 0.0, 100.0))

    # Build diagnostic boundary visualization RGB
    vis_rgb = rectified_rgb.copy()
    # Draw region partition lines lightly
    for ebox in rmap.edge_regions.values():
        cv2.rectangle(vis_rgb, (ebox.x1, ebox.y1), (ebox.x2 - 1, ebox.y2 - 1), (80, 200, 120), 1)
    for cbox in rmap.corner_regions.values():
        cv2.rectangle(vis_rgb, (cbox.x1, cbox.y1), (cbox.x2 - 1, cbox.y2 - 1), (60, 160, 240), 1)

    # Highlight missing edge/corner defects in bright red
    defect_union = (edge_wear_mask > 0) | (corner_damage_mask > 0)
    vis_rgb[defect_union] = [235, 45, 45]

    warnings: List[str] = []
    if not segmentation_succeeded:
        warnings.append(
            "Automatic note boundary segmentation did not isolate an external background; "
            "edge and corner wear scores reflect only visible border deficits."
        )

    worst_edge = max(edge_metrics.values(), key=lambda x: x.edge_wear_score)
    worst_corner = max(corner_metrics.values(), key=lambda x: x.corner_damage_score)
    explanation = (
        f"Edge condition score: {overall_edge_cond:.1f}/100 (highest wear on '{worst_edge.edge_name}' "
        f"edge: {worst_edge.edge_wear_score:.1f}%). "
        f"Corner condition score: {overall_corner_cond:.1f}/100 (highest damage on "
        f"'{worst_corner.corner_name}' corner: {worst_corner.corner_damage_score:.1f}%)."
    )

    return EdgeAnalysisResult(
        edge_metrics=edge_metrics,
        corner_metrics=corner_metrics,
        overall_edge_wear_score=overall_edge_wear,
        overall_edge_condition_score=overall_edge_cond,
        overall_corner_damage_score=overall_corner_dmg,
        overall_corner_condition_score=overall_corner_cond,
        internal_edge_density=internal_edge_density,
        mean_sobel_gradient=mean_sobel,
        canny_edges=canny_edges,
        sobel_magnitude_vis=sobel_vis,
        edge_wear_mask=edge_wear_mask,
        corner_damage_mask=corner_damage_mask,
        boundary_visualization_rgb=vis_rgb,
        is_boundary_assessable=True,
        explanation=explanation,
        warnings=warnings,
    )
