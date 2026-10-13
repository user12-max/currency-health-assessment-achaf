"""Unit tests for Module 2: Capture Quality and Image Preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

from config.settings import AnalysisSettings
from src.evaluation import generate_synthetic_test_note
from src.image_quality import assess_capture_quality
from src.preprocessing import (
    apply_clahe_enhancement,
    apply_denoising,
    normalize_illumination,
    preprocess_image,
    resize_preserving_aspect,
)


def test_resize_preserving_aspect() -> None:
    """Verify aspect ratio is preserved when downscaling large images."""
    img = np.zeros((800, 1600, 3), dtype=np.uint8)
    img[::2, ::2] = 200
    resized, sx, sy = resize_preserving_aspect(img, max_dimension=800)
    assert resized.shape == (400, 800, 3)
    assert np.isclose(sx, 0.5)
    assert np.isclose(sy, 0.5)

    # Image smaller than max_dimension remains unchanged
    small, sx2, sy2 = resize_preserving_aspect(resized, max_dimension=1000)
    assert small.shape == (400, 800, 3)
    assert sx2 == 1.0 and sy2 == 1.0


def test_capture_quality_blur_and_exposure_detection() -> None:
    """Verify sharp vs blurry and underexposed vs overexposed images are flagged accurately."""
    sharp_note, _ = generate_synthetic_test_note(variant="clean")
    q_sharp = assess_capture_quality(sharp_note)
    assert q_sharp.is_blurry is False
    assert q_sharp.is_severely_blurry is False
    assert q_sharp.is_provisional is False
    assert q_sharp.quality_rating in ("Good", "Acceptable")

    # Apply strong Gaussian blur to simulate out-of-focus camera capture
    blurry_note = cv2.GaussianBlur(sharp_note, (31, 31), 9.0)
    q_blur = assess_capture_quality(blurry_note)
    assert q_blur.is_blurry is True
    assert q_blur.is_severely_blurry is True
    assert q_blur.is_provisional is True
    assert any("blur" in w.lower() for w in q_blur.warnings)

    # Simulate severe underexposure
    dark_note = (sharp_note.astype(np.float32) * 0.08).astype(np.uint8)
    q_dark = assess_capture_quality(dark_note)
    assert q_dark.is_underexposed is True

    # Simulate severe overexposure
    bright_note = np.clip(sharp_note.astype(np.float32) * 1.8 + 140.0, 0, 255).astype(np.uint8)
    q_bright = assess_capture_quality(bright_note)
    assert q_bright.is_overexposed is True


def test_denoising_and_illumination_and_clahe() -> None:
    """Verify denoising modes, illumination normalization, and CLAHE execute cleanly."""
    note, _ = generate_synthetic_test_note(variant="damaged")
    for method in ("bilateral", "gaussian", "median", "none"):
        den = apply_denoising(note, method=method, strength=5)
        assert den.shape == note.shape
        assert den.dtype == np.uint8

    illum = normalize_illumination(note)
    assert illum.shape == note.shape
    assert illum.dtype == np.uint8

    enh_rgb, clahe_g = apply_clahe_enhancement(note, clip_limit=2.0, tile_grid_size=8)
    assert enh_rgb.shape == note.shape
    assert clahe_g.shape == note.shape[:2]


def test_full_preprocess_image_pipeline() -> None:
    """Verify preprocess_image returns all required color spaces and quality metrics."""
    note, _ = generate_synthetic_test_note(variant="clean")
    cfg = AnalysisSettings(target_max_width=800)
    res = preprocess_image(note, settings=cfg)

    assert res.resized_rgb.shape[1] <= 800
    assert res.gray.ndim == 2
    assert res.hsv.shape == res.resized_rgb.shape
    assert res.lab.shape == res.resized_rgb.shape
    assert res.quality_report.width == note.shape[1]
