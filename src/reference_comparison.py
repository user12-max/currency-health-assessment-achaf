"""Module 9: Reference-Assisted Comparison and Alignment Validation.

Aligns a clean reference currency note image to the rectified test note using
ORB local feature descriptors, Lowe's ratio test, and RANSAC homography
estimation. Performs strict geometric and photometric validation to ensure
registration is reliable before computing region-wise structural, texture, and
color differences.

If alignment is unreliable (insufficient inliers, degenerate homography, or
incompatible design/side), registration is explicitly rejected and the pipeline
falls back safely to single-image feature-based analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from config.settings import AnalysisSettings
from src.preprocessing import normalize_illumination


@dataclass
class ReferenceComparisonResult:
    """Structured result of ORB+RANSAC reference registration and difference analysis."""

    reference_supplied: bool
    alignment_succeeded: bool
    keypoints_test_count: int = 0
    keypoints_ref_count: int = 0
    good_matches_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    homography_det: float = 0.0
    global_ssim: float = 0.0
    aligned_reference_rgb: Optional[np.ndarray] = None
    aligned_reference_gray: Optional[np.ndarray] = None
    structural_diff_map: Optional[np.ndarray] = None   # uint8 [0..255]
    color_diff_map: Optional[np.ndarray] = None        # uint8 [0..255]
    combined_diff_mask: Optional[np.ndarray] = None    # Binary uint8 mask (255 = significant diff)
    match_visualization_rgb: Optional[np.ndarray] = None
    rejection_reason: Optional[str] = None
    status_message: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, object]:
        """Return JSON-serializable registration summary."""
        return {
            "reference_supplied": self.reference_supplied,
            "alignment_succeeded": self.alignment_succeeded,
            "keypoints_test_count": self.keypoints_test_count,
            "keypoints_ref_count": self.keypoints_ref_count,
            "good_matches_count": self.good_matches_count,
            "inlier_count": self.inlier_count,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "homography_det": round(self.homography_det, 4),
            "global_ssim": round(self.global_ssim, 4),
            "rejection_reason": self.rejection_reason,
            "status_message": self.status_message,
            "warnings": list(self.warnings),
        }


def _validate_homography_geometry(
    H: np.ndarray, ref_shape: Tuple[int, int], dst_shape: Tuple[int, int]
) -> Tuple[bool, float, Optional[str]]:
    """Verify that a 3x3 homography matrix is non-degenerate and preserves convex quad geometry."""
    if H is None or H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return False, 0.0, "Homography matrix is None or contains non-finite values."

    # Normalize by H[2, 2]
    if abs(float(H[2, 2])) < 1e-8:
        return False, 0.0, "Homography scale factor H[2,2] is near zero."
    H_norm = H / float(H[2, 2])

    det = float(np.linalg.det(H_norm[:2, :2]))
    if det < 0.15 or det > 6.5:
        return (
            False,
            det,
            f"Implausible geometric scaling/reflection (determinant={det:.2f} outside [0.15, 6.5]).",
        )

    # Check perspective warp terms (H[2, 0], H[2, 1]) are not extreme
    if abs(float(H_norm[2, 0])) > 0.004 or abs(float(H_norm[2, 1])) > 0.004:
        return (
            False,
            det,
            "Extreme projective distortion detected in homography matrix.",
        )

    ref_h, ref_w = ref_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    corners = np.array(
        [[[0, 0]], [[ref_w - 1, 0]], [[ref_w - 1, ref_h - 1]], [[0, ref_h - 1]]],
        dtype=np.float32,
    )
    warped_corners = cv2.perspectiveTransform(corners, H_norm).reshape(4, 2)

    if not cv2.isContourConvex(warped_corners.astype(np.float32)):
        return False, det, "Warped reference corners do not form a convex quadrilateral."

    area = float(cv2.contourArea(warped_corners.astype(np.float32)))
    dst_area = float(dst_w * dst_h)
    area_ratio = area / max(dst_area, 1.0)
    if area_ratio < 0.45 or area_ratio > 1.85:
        return (
            False,
            det,
            f"Warped reference area ratio ({area_ratio:.2f}) is incompatible with test note dimensions.",
        )

    return True, det, None


def register_and_compare_reference(
    test_rectified_rgb: np.ndarray,
    reference_rectified_rgb: Optional[np.ndarray],
    test_mask: Optional[np.ndarray] = None,
    settings: Optional[AnalysisSettings] = None,
) -> ReferenceComparisonResult:
    """Align reference note to test note via ORB + RANSAC and compute difference maps.

    Parameters
    ----------
    test_rectified_rgb : np.ndarray
        Rectified RGB uint8 image of the test note.
    reference_rectified_rgb : np.ndarray, optional
        Rectified RGB uint8 image of the clean reference note.
    test_mask : np.ndarray, optional
        Foreground binary mask of the test note.
    settings : AnalysisSettings, optional
        Pipeline configuration parameters.
    """
    if reference_rectified_rgb is None or reference_rectified_rgb.size == 0:
        return ReferenceComparisonResult(
            reference_supplied=False,
            alignment_succeeded=False,
            status_message="No reference image supplied; operating in single-image feature-based mode.",
        )

    cfg = settings or AnalysisSettings()
    dst_h, dst_w = test_rectified_rgb.shape[:2]

    # Resize reference to canonical dimensions before feature extraction so scales are comparable
    ref_resized = cv2.resize(
        reference_rectified_rgb, (dst_w, dst_h), interpolation=cv2.INTER_AREA
    )

    # Illumination normalization on both images prior to registration and comparison
    test_norm_rgb = normalize_illumination(test_rectified_rgb)
    ref_norm_rgb = normalize_illumination(ref_resized)

    test_gray = cv2.cvtColor(test_norm_rgb, cv2.COLOR_RGB2GRAY)
    ref_gray = cv2.cvtColor(ref_norm_rgb, cv2.COLOR_RGB2GRAY)

    # CLAHE for robust feature detection
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    test_kp_img = clahe.apply(test_gray)
    ref_kp_img = clahe.apply(ref_gray)

    orb = cv2.ORB_create(nfeatures=max(500, int(cfg.orb_max_features)))
    kp_test, desc_test = orb.detectAndCompute(test_kp_img, None)
    kp_ref, desc_ref = orb.detectAndCompute(ref_kp_img, None)

    n_kp_test = len(kp_test) if kp_test is not None else 0
    n_kp_ref = len(kp_ref) if kp_ref is not None else 0

    if (
        desc_test is None
        or desc_ref is None
        or n_kp_test < cfg.min_inlier_matches
        or n_kp_ref < cfg.min_inlier_matches
    ):
        reason = (
            f"Insufficient ORB keypoints detected (test={n_kp_test}, reference={n_kp_ref}; "
            f"minimum required={cfg.min_inlier_matches})."
        )
        return ReferenceComparisonResult(
            reference_supplied=True,
            alignment_succeeded=False,
            keypoints_test_count=n_kp_test,
            keypoints_ref_count=n_kp_ref,
            rejection_reason=reason,
            status_message=f"Reference registration rejected: {reason} Falling back to feature-based mode.",
            warnings=[reason],
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw_matches = matcher.knnMatch(desc_ref, desc_test, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.78 * n.distance:
                good_matches.append(m)

    n_good = len(good_matches)
    if n_good < cfg.min_inlier_matches:
        reason = (
            f"Insufficient descriptor matches passed Lowe's ratio test "
            f"({n_good} < {cfg.min_inlier_matches}). Ensure the reference image shows "
            "the same denomination, side, and design."
        )
        return ReferenceComparisonResult(
            reference_supplied=True,
            alignment_succeeded=False,
            keypoints_test_count=n_kp_test,
            keypoints_ref_count=n_kp_ref,
            good_matches_count=n_good,
            rejection_reason=reason,
            status_message=f"Reference registration rejected: {reason} Falling back to feature-based mode.",
            warnings=[reason],
        )

    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransacReprojThreshold=float(cfg.ransac_reproj_threshold),
    )

    if H is None or mask is None:
        reason = "RANSAC homography estimation failed to converge."
        return ReferenceComparisonResult(
            reference_supplied=True,
            alignment_succeeded=False,
            keypoints_test_count=n_kp_test,
            keypoints_ref_count=n_kp_ref,
            good_matches_count=n_good,
            rejection_reason=reason,
            status_message=f"Reference registration rejected: {reason} Falling back to feature-based mode.",
            warnings=[reason],
        )

    inliers_mask = mask.ravel().astype(bool)
    inlier_count = int(np.sum(inliers_mask))
    inlier_ratio = float(inlier_count) / float(max(1, n_good))

    if inlier_count < cfg.min_inlier_matches or inlier_ratio < cfg.min_inlier_ratio:
        reason = (
            f"RANSAC alignment failed reliability thresholds (inliers={inlier_count}, "
            f"inlier ratio={inlier_ratio * 100:.1f}%; required >= {cfg.min_inlier_matches} "
            f"inliers and >= {cfg.min_inlier_ratio * 100:.0f}% ratio)."
        )
        return ReferenceComparisonResult(
            reference_supplied=True,
            alignment_succeeded=False,
            keypoints_test_count=n_kp_test,
            keypoints_ref_count=n_kp_ref,
            good_matches_count=n_good,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rejection_reason=reason,
            status_message=f"Reference registration rejected: {reason} Falling back to feature-based mode.",
            warnings=[reason],
        )

    geom_ok, det, geom_err = _validate_homography_geometry(
        H, (dst_h, dst_w), (dst_h, dst_w)
    )
    if not geom_ok:
        reason = geom_err or "Degenerate homography geometry."
        return ReferenceComparisonResult(
            reference_supplied=True,
            alignment_succeeded=False,
            keypoints_test_count=n_kp_test,
            keypoints_ref_count=n_kp_ref,
            good_matches_count=n_good,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            homography_det=det,
            rejection_reason=reason,
            status_message=f"Reference registration rejected: {reason} Falling back to feature-based mode.",
            warnings=[reason],
        )

    # Warp reference RGB and grayscale into test note coordinate frame
    aligned_ref_rgb = cv2.warpPerspective(
        ref_resized,
        H,
        (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    aligned_ref_norm_rgb = normalize_illumination(aligned_ref_rgb)
    aligned_ref_gray = cv2.cvtColor(aligned_ref_norm_rgb, cv2.COLOR_RGB2GRAY)

    # Compute structural similarity (SSIM) on lightly smoothed grayscale to ignore sub-pixel jitter
    test_smooth = cv2.GaussianBlur(test_gray, (5, 5), 1.0)
    ref_smooth = cv2.GaussianBlur(aligned_ref_gray, (5, 5), 1.0)

    ssim_score, ssim_map = ssim(
        test_smooth, ref_smooth, data_range=255, full=True
    )
    global_ssim_val = float(ssim_score)

    if global_ssim_val < 0.22:
        reason = (
            f"Post-alignment structural similarity is too low (SSIM={global_ssim_val:.2f} < 0.22). "
            "The reference image likely represents a different denomination, side, or series."
        )
        return ReferenceComparisonResult(
            reference_supplied=True,
            alignment_succeeded=False,
            keypoints_test_count=n_kp_test,
            keypoints_ref_count=n_kp_ref,
            good_matches_count=n_good,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            homography_det=det,
            global_ssim=global_ssim_val,
            rejection_reason=reason,
            status_message=f"Reference registration rejected: {reason} Falling back to feature-based mode.",
            warnings=[reason],
        )

    # Build structural difference map (1 - SSIM) and CIE Lab color difference map
    struct_diff = np.clip((1.0 - ssim_map) * 255.0, 0, 255).astype(np.uint8)

    test_lab = cv2.cvtColor(test_norm_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(aligned_ref_norm_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    test_lab_s = cv2.GaussianBlur(test_lab, (7, 7), 0)
    ref_lab_s = cv2.GaussianBlur(ref_lab, (7, 7), 0)
    delta_e = np.linalg.norm(test_lab_s - ref_lab_s, axis=2)
    color_diff = np.clip((delta_e / 60.0) * 255.0, 0, 255).astype(np.uint8)

    combined_diff = np.clip(
        0.55 * struct_diff.astype(np.float32) + 0.45 * color_diff.astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    if test_mask is not None and test_mask.shape[:2] == (dst_h, dst_w):
        combined_diff = cv2.bitwise_and(combined_diff, test_mask)

    _, combined_diff_mask = cv2.threshold(combined_diff, 105, 255, cv2.THRESH_BINARY)
    morph_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_diff_mask = cv2.morphologyEx(combined_diff_mask, cv2.MORPH_OPEN, morph_k)

    # Draw inlier keypoint matches for visual inspection
    inlier_matches = [m for m, is_in in zip(good_matches, inliers_mask) if is_in][:60]
    match_vis_bgr = cv2.drawMatches(
        cv2.cvtColor(ref_resized, cv2.COLOR_RGB2BGR),
        kp_ref,
        cv2.cvtColor(test_rectified_rgb, cv2.COLOR_RGB2BGR),
        kp_test,
        inlier_matches,
        None,
        matchColor=(40, 220, 90),
        singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    match_vis_rgb = cv2.cvtColor(match_vis_bgr, cv2.COLOR_BGR2RGB)

    warnings: List[str] = []
    if global_ssim_val < 0.45:
        warnings.append(
            f"Moderate alignment similarity (SSIM={global_ssim_val:.2f}). Minor registration "
            "offsets or lighting differences may affect fine-scale reference comparisons."
        )

    return ReferenceComparisonResult(
        reference_supplied=True,
        alignment_succeeded=True,
        keypoints_test_count=n_kp_test,
        keypoints_ref_count=n_kp_ref,
        good_matches_count=n_good,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        homography_det=det,
        global_ssim=global_ssim_val,
        aligned_reference_rgb=aligned_ref_rgb,
        aligned_reference_gray=aligned_ref_gray,
        structural_diff_map=struct_diff,
        color_diff_map=color_diff,
        combined_diff_mask=combined_diff_mask,
        match_visualization_rgb=match_vis_rgb,
        rejection_reason=None,
        status_message=(
            f"Reference alignment succeeded ({inlier_count} RANSAC inliers, "
            f"inlier ratio={inlier_ratio * 100:.1f}%, SSIM={global_ssim_val:.2f})."
        ),
        warnings=warnings,
    )
