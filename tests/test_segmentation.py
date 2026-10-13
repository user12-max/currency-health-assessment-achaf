"""Unit tests for Module 3: Currency Note Segmentation and Perspective Correction."""

from __future__ import annotations

import numpy as np

from config.settings import AnalysisSettings, STANDARD_RECTIFIED_WIDTH
from src.evaluation import generate_synthetic_test_note
from src.segmentation import (
    apply_manual_crop,
    order_quad_points,
    segment_and_rectify_note,
)


def test_order_quad_points_consistency() -> None:
    """Verify 4 scrambled corner points are ordered as TL, TR, BR, BL."""
    scrambled = np.array(
        [[700.0, 350.0], [50.0, 40.0], [55.0, 360.0], [690.0, 35.0]],
        dtype=np.float32,
    )
    ordered = order_quad_points(scrambled)
    np.testing.assert_allclose(ordered[0], [50.0, 40.0])   # TL
    np.testing.assert_allclose(ordered[1], [690.0, 35.0])  # TR
    np.testing.assert_allclose(ordered[2], [700.0, 350.0]) # BR
    np.testing.assert_allclose(ordered[3], [55.0, 360.0])  # BL


def test_segmentation_success_on_clean_and_angled_notes() -> None:
    """Verify automatic segmentation isolates and rectifies both flat and angled notes."""
    for variant in ("clean", "damaged", "angled_damaged"):
        img, _ = generate_synthetic_test_note(variant=variant)
        res = segment_and_rectify_note(img)
        assert res.segmentation_succeeded is True
        assert res.perspective_corrected is True
        assert res.rectified_rgb.shape[1] == STANDARD_RECTIFIED_WIDTH
        assert res.rectified_mask.shape == res.rectified_rgb.shape[:2]
        assert res.homography is not None and res.homography.shape == (3, 3)
        assert res.inv_homography is not None and res.inv_homography.shape == (3, 3)
        # Forward * inverse homography should approximate identity matrix
        prod = res.homography @ res.inv_homography
        prod /= prod[2, 2]
        np.testing.assert_allclose(prod, np.eye(3), atol=1e-3)


def test_segmentation_failure_and_manual_fallback() -> None:
    """Verify segmentation fails honestly when no valid note contour exists and falls back."""
    # Create image with tiny speck in center (< 15% minimum area ratio)
    bg = np.full((400, 800, 3), 30, dtype=np.uint8)
    bg[190:210, 380:420] = 220

    res = segment_and_rectify_note(bg)
    assert res.segmentation_succeeded is False
    assert res.perspective_corrected is False
    assert res.method_used == "manual_crop_fallback"
    assert len(res.warnings) > 0

    # Test explicit manual crop margins
    cfg = AnalysisSettings(
        enable_auto_segmentation=False,
        manual_crop_margins=(0.10, 0.10, 0.10, 0.10),
    )
    manual_res = segment_and_rectify_note(bg, settings=cfg)
    assert manual_res.segmentation_succeeded is False
    assert manual_res.method_used == "manual_crop_fallback"

    direct_crop = apply_manual_crop(bg, margins=(0.05, 0.05, 0.05, 0.05))
    assert direct_crop.rectified_rgb.shape[1] == STANDARD_RECTIFIED_WIDTH
