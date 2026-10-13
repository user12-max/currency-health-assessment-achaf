"""Module 3: Currency Note Segmentation and Perspective Correction.

Isolates the currency note from its background using multi-cue foreground
extraction (Canny edges, Otsu/adaptive thresholding, border background modeling,
contour extraction, and quadrilateral approximation), performs 4-point
perspective rectification, and preserves both forward and inverse homography
matrices for accurate spatial localization.

When automatic segmentation cannot reliably isolate a quadrilateral contour,
the module explicitly reports segmentation failure and falls back to manual
margin cropping or full-frame analysis without pretending perspective correction
succeeded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    AnalysisSettings,
    STANDARD_RECTIFIED_HEIGHT,
    STANDARD_RECTIFIED_WIDTH,
)


@dataclass
class SegmentationResult:
    """Result of currency-note segmentation and perspective rectification."""

    segmentation_succeeded: bool
    perspective_corrected: bool
    method_used: str  # "quad_perspective", "min_area_rect", "already_cropped", "manual_crop_fallback"
    rectified_rgb: np.ndarray
    rectified_mask: np.ndarray  # Binary mask (255 = note foreground, 0 = background/missing)
    original_contour_mask: np.ndarray  # Foreground mask in resized input coordinates
    detected_quad_corners: Optional[np.ndarray] = None  # Shape (4, 2) float32 in resized coords
    detected_contour: Optional[np.ndarray] = None
    homography: Optional[np.ndarray] = None      # 3x3 transform: resized -> rectified
    inv_homography: Optional[np.ndarray] = None  # 3x3 transform: rectified -> resized
    contour_area_ratio: float = 1.0
    estimated_aspect_ratio: float = 2.1
    status_message: str = ""
    warnings: List[str] = field(default_factory=list)


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points consistently as: top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    # Sort by y-coordinate (top two vs bottom two)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_pts = y_sorted[:2]
    bottom_pts = y_sorted[2:]

    tl, tr = top_pts[np.argsort(top_pts[:, 0])]
    bl, br = bottom_pts[np.argsort(bottom_pts[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _compute_foreground_mask(
    rgb_image: np.ndarray,
    canny_low: int = 40,
    canny_high: int = 140,
    alpha_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Combine border-background modeling, Otsu thresholding, and Canny edges to extract note mask."""
    h, w = rgb_image.shape[:2]

    # 1. If explicit alpha mask exists and has transparent background, use it directly
    if alpha_mask is not None and alpha_mask.shape[:2] == (h, w):
        if np.mean(alpha_mask < 128) > 0.03:
            mask = np.where(alpha_mask >= 128, 255, 0).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 2. Estimate background color distribution from a thin outer frame (2% border)
    b_h = max(2, int(round(h * 0.025)))
    b_w = max(2, int(round(w * 0.025)))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:b_h, :] = True
    border_mask[-b_h:, :] = True
    border_mask[:, :b_w] = True
    border_mask[:, -b_w:] = True

    border_pixels_lab = lab[border_mask]
    bg_median_lab = np.median(border_pixels_lab, axis=0)
    border_std = float(np.mean(np.std(border_pixels_lab, axis=0)))

    # Color distance from border background in CIE Lab space
    diff_lab = np.linalg.norm(lab - bg_median_lab[np.newaxis, np.newaxis, :], axis=2)

    # Adaptive threshold on color distance if border is reasonably uniform
    if border_std < 28.0:
        dist_thresh = max(16.0, min(42.0, border_std * 2.2 + 12.0))
        bg_diff_mask = np.where(diff_lab > dist_thresh, 255, 0).astype(np.uint8)
    else:
        bg_diff_mask = np.zeros((h, w), dtype=np.uint8)

    # 3. Otsu thresholding (choose polarity that places center of image as foreground)
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    center_patch = otsu_mask[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    border_patch_mean = float(np.mean(otsu_mask[border_mask]))
    if float(np.mean(center_patch)) < border_patch_mean:
        otsu_mask = cv2.bitwise_not(otsu_mask)

    # 4. Canny edges + morphological closing
    edges = cv2.Canny(blurred, canny_low, canny_high)
    close_k = max(7, int(round(min(h, w) * 0.03)))
    if close_k % 2 == 0:
        close_k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    edge_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Fill external contours of closed edges
    cnts, _ = cv2.findContours(edge_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    edge_filled = np.zeros((h, w), dtype=np.uint8)
    if cnts:
        largest_edge_cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest_edge_cnt) > 0.12 * (h * w):
            cv2.drawContours(edge_filled, [largest_edge_cnt], -1, 255, thickness=cv2.FILLED)

    # Combine cues: prefer bg_diff_mask when strong, reinforced by otsu & edge_filled
    if np.mean(bg_diff_mask > 0) > 0.15 and np.mean(bg_diff_mask > 0) < 0.96:
        combined = cv2.bitwise_or(bg_diff_mask, edge_filled)
    else:
        combined = cv2.bitwise_or(otsu_mask, edge_filled)

    # Clean up mask with morphological closing then opening
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    open_k = max(3, close_k // 2)
    if open_k % 2 == 0:
        open_k += 1
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel)

    return combined


def _fit_quadrilateral_envelope(contour: np.ndarray) -> Tuple[np.ndarray, str]:
    """Fit a 4-corner quadrilateral envelope around a currency note contour.

    Uses convex hull polygon approximation first; if a clean 4-vertex polygon is
    not found or is geometrically skewed, uses the minimum-area bounding rectangle
    so that missing corners or edge tears remain inside the quadrilateral frame
    and can be measured as foreground deficits.
    """
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)

    for eps_factor in (0.015, 0.025, 0.035, 0.05, 0.07):
        approx = cv2.approxPolyDP(hull, eps_factor * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = order_quad_points(approx.reshape(4, 2))
            # Check that quad area is close to minAreaRect area (if a corner is torn off,
            # minAreaRect preserves the full expected corner while approxPolyDP might cut it)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            rect_area = max(1.0, float(cv2.contourArea(box)))
            quad_area = float(cv2.contourArea(quad))
            if 0.92 <= (quad_area / rect_area) <= 1.05:
                return quad, "quad_perspective"

    # Fallback to minimum-area bounding rectangle of the contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_quad_points(box), "min_area_rect"


def _rectify_from_quad(
    rgb_image: np.ndarray,
    contour_mask: np.ndarray,
    quad_pts: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Warp image and foreground mask using 4 ordered quad corners into a canonical frame."""
    quad = order_quad_points(quad_pts)
    tl, tr, br, bl = quad

    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    max_w = max(width_top, width_bottom, 1.0)

    height_left = float(np.linalg.norm(bl - tl))
    height_right = float(np.linalg.norm(br - tr))
    max_h = max(height_left, height_right, 1.0)

    # If the note is oriented vertically (portrait), rotate quad ordering so width > height
    if max_h > max_w * 1.15:
        quad = np.array([bl, tl, tr, br], dtype=np.float32)
        max_w, max_h = max_h, max_w

    aspect_ratio = float(max_w / max(max_h, 1.0))

    dst_w = STANDARD_RECTIFIED_WIDTH
    # Clamp target height to reasonable bounds while respecting note aspect ratio
    computed_h = int(round(dst_w / np.clip(aspect_ratio, 1.4, 2.8)))
    dst_h = max(240, min(560, computed_h))

    dst_pts = np.array(
        [
            [0.0, 0.0],
            [float(dst_w - 1), 0.0],
            [float(dst_w - 1), float(dst_h - 1)],
            [0.0, float(dst_h - 1)],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(quad, dst_pts)
    inv_homography = cv2.getPerspectiveTransform(dst_pts, quad)

    rectified_rgb = cv2.warpPerspective(
        rgb_image,
        homography,
        (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    rectified_mask = cv2.warpPerspective(
        contour_mask,
        homography,
        (dst_w, dst_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rectified_mask = np.where(rectified_mask >= 128, 255, 0).astype(np.uint8)

    return rectified_rgb, rectified_mask, homography, inv_homography, aspect_ratio


def apply_manual_crop(
    rgb_image: np.ndarray,
    margins: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> SegmentationResult:
    """Apply user-specified relative crop margins (top, bottom, left, right in [0, 0.45])."""
    h, w = rgb_image.shape[:2]
    top_m, bot_m, left_m, right_m = [float(np.clip(m, 0.0, 0.45)) for m in margins]

    y1 = int(round(h * top_m))
    y2 = max(y1 + 20, int(round(h * (1.0 - bot_m))))
    x1 = int(round(w * left_m))
    x2 = max(x1 + 40, int(round(w * (1.0 - right_m))))

    quad = np.array(
        [[x1, y1], [x2 - 1, y1], [x2 - 1, y2 - 1], [x1, y2 - 1]],
        dtype=np.float32,
    )
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = 255

    rect_rgb, rect_mask, H, H_inv, aspect = _rectify_from_quad(
        rgb_image, full_mask, quad
    )

    return SegmentationResult(
        segmentation_succeeded=False,
        perspective_corrected=False,
        method_used="manual_crop_fallback",
        rectified_rgb=rect_rgb,
        rectified_mask=rect_mask,
        original_contour_mask=full_mask,
        detected_quad_corners=quad,
        detected_contour=quad.reshape(-1, 1, 2).astype(np.int32),
        homography=H,
        inv_homography=H_inv,
        contour_area_ratio=float((y2 - y1) * (x2 - x1)) / float(max(1, h * w)),
        estimated_aspect_ratio=aspect,
        status_message=(
            "Manual crop / full-frame fallback applied. Automatic perspective "
            "rectification was bypassed or did not find a distinct background boundary."
        ),
        warnings=[
            "Operating in manual/cropped fallback mode: external edge and corner wear "
            "can only be estimated if the true note borders remain visible inside the frame."
        ],
    )


def segment_and_rectify_note(
    rgb_image: np.ndarray,
    settings: Optional[AnalysisSettings] = None,
    alpha_mask: Optional[np.ndarray] = None,
) -> SegmentationResult:
    """Segment the currency note from the background and rectify its perspective.

    Parameters
    ----------
    rgb_image : np.ndarray
        Preprocessed (resized) RGB uint8 image.
    settings : AnalysisSettings, optional
        Segmentation configuration parameters.
    alpha_mask : np.ndarray, optional
        Optional alpha channel mask resized to match `rgb_image`.

    Returns
    -------
    SegmentationResult
        Rectified note image, foreground mask, homographies, and diagnostic status.
    """
    cfg = settings or AnalysisSettings()
    h, w = rgb_image.shape[:2]

    # Check if user explicitly disabled auto-segmentation or requested manual crop
    if not cfg.enable_auto_segmentation or any(m > 0.001 for m in cfg.manual_crop_margins):
        return apply_manual_crop(rgb_image, cfg.manual_crop_margins)

    fg_mask = _compute_foreground_mask(
        rgb_image,
        canny_low=cfg.canny_low_threshold,
        canny_high=cfg.canny_high_threshold,
        alpha_mask=alpha_mask,
    )

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        res = apply_manual_crop(rgb_image, (0.0, 0.0, 0.0, 0.0))
        res.status_message = (
            "Automatic segmentation failed: no foreground contour detected. "
            "Falling back to unrectified full-frame image."
        )
        return res

    largest_cnt = max(contours, key=cv2.contourArea)
    cnt_area = float(cv2.contourArea(largest_cnt))
    img_area = float(h * w)
    area_ratio = cnt_area / max(img_area, 1.0)

    if area_ratio < cfg.min_note_area_ratio:
        res = apply_manual_crop(rgb_image, (0.0, 0.0, 0.0, 0.0))
        res.status_message = (
            f"Automatic segmentation failed: largest contour occupies only "
            f"{area_ratio * 100:.1f}% of the frame (minimum required: "
            f"{cfg.min_note_area_ratio * 100:.0f}%). Using manual/full-frame fallback."
        )
        return res

    # Build clean single-component contour mask
    clean_cnt_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean_cnt_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)

    quad_pts, fit_method = _fit_quadrilateral_envelope(largest_cnt)
    rect_rgb, rect_mask, H, H_inv, aspect = _rectify_from_quad(
        rgb_image, clean_cnt_mask, quad_pts
    )

    warnings: List[str] = []
    if aspect < 1.30 or aspect > 3.10:
        warnings.append(
            f"Detected contour aspect ratio ({aspect:.2f}:1) deviates from typical "
            "currency note proportions (~1.9:1 to 2.3:1). Verify that the note is "
            "fully visible and not occluded."
        )

    # Check if the contour covers almost the entire image frame (>= 97%)
    if area_ratio >= 0.97:
        return SegmentationResult(
            segmentation_succeeded=True,
            perspective_corrected=False,
            method_used="already_cropped",
            rectified_rgb=rect_rgb,
            rectified_mask=rect_mask,
            original_contour_mask=clean_cnt_mask,
            detected_quad_corners=quad_pts,
            detected_contour=largest_cnt,
            homography=H,
            inv_homography=H_inv,
            contour_area_ratio=area_ratio,
            estimated_aspect_ratio=aspect,
            status_message=(
                "Input image appears tightly cropped to the currency note boundary "
                f"(occupies {area_ratio * 100:.1f}% of frame). Standardized to canonical dimensions."
            ),
            warnings=warnings,
        )

    return SegmentationResult(
        segmentation_succeeded=True,
        perspective_corrected=True,
        method_used=fit_method,
        rectified_rgb=rect_rgb,
        rectified_mask=rect_mask,
        original_contour_mask=clean_cnt_mask,
        detected_quad_corners=quad_pts,
        detected_contour=largest_cnt,
        homography=H,
        inv_homography=H_inv,
        contour_area_ratio=area_ratio,
        estimated_aspect_ratio=aspect,
        status_message=(
            f"Note boundary segmented ({area_ratio * 100:.1f}% frame area) and "
            f"perspective rectified via {fit_method}."
        ),
        warnings=warnings,
    )
