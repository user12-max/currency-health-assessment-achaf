"""Unit tests for Module 1: Image Acquisition and Validation (`src.image_io`)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.image_io import load_and_validate_image


def _make_pil_bytes(img: Image.Image, fmt: str = "PNG", exif_orientation: int = 1) -> bytes:
    buf = io.BytesIO()
    if exif_orientation != 1:
        exif = img.getexif()
        exif[0x0112] = exif_orientation
        img.save(buf, format=fmt, exif=exif.tobytes())
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def test_valid_rgb_png_and_jpeg_bytes() -> None:
    """Verify valid RGB PNG and JPEG images load with accurate dimensions and channels."""
    rng = np.random.default_rng(10)
    arr = rng.integers(30, 220, size=(240, 500, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr, mode="RGB")

    for fmt in ("PNG", "JPEG", "WEBP"):
        raw = _make_pil_bytes(pil_img, fmt=fmt)
        res = load_and_validate_image(raw)
        assert res.is_valid is True
        assert res.rgb_image is not None
        assert res.bgr_image is not None
        assert res.width == 500
        assert res.height == 240
        assert res.channels == 3
        assert res.rgb_image.shape == (240, 500, 3)
        # Verify RGB and BGR channel swap consistency
        np.testing.assert_array_equal(res.rgb_image[:, :, 0], res.bgr_image[:, :, 2])


def test_grayscale_image_handling() -> None:
    """Verify single-channel grayscale images load safely and emit an informative warning."""
    rng = np.random.default_rng(11)
    gray = rng.integers(20, 230, size=(220, 440), dtype=np.uint8)
    pil_gray = Image.fromarray(gray, mode="L")
    raw = _make_pil_bytes(pil_gray, fmt="PNG")

    res = load_and_validate_image(raw)
    assert res.is_valid is True
    assert res.original_mode == "L"
    assert res.rgb_image is not None
    assert res.rgb_image.shape == (220, 440, 3)
    assert any("grayscale" in w.lower() for w in res.warnings)


def test_transparent_rgba_image_handling() -> None:
    """Verify RGBA images with transparency preserve alpha_mask and composite cleanly."""
    rgba = np.full((220, 440, 4), 180, dtype=np.uint8)
    rgba[:, :50, 3] = 0  # Left strip transparent
    rgba[50:150, 100:300, 0] = 90  # Add variance so image is not uniform
    pil_rgba = Image.fromarray(rgba, mode="RGBA")
    raw = _make_pil_bytes(pil_rgba, fmt="PNG")

    res = load_and_validate_image(raw)
    assert res.is_valid is True
    assert res.has_transparency is True
    assert res.alpha_mask is not None
    assert res.alpha_mask.shape == (220, 440)
    assert np.all(res.alpha_mask[:, :50] == 0)
    assert np.all(res.alpha_mask[:, 60:] == 180)
    assert any("transparency" in w.lower() for w in res.warnings)


def test_exif_orientation_transposition() -> None:
    """Verify EXIF orientation tag (e.g., 6 = Rotate 90 CW) transposes image dimensions."""
    rng = np.random.default_rng(12)
    arr = rng.integers(20, 230, size=(200, 420, 3), dtype=np.uint8)
    pil_img = Image.fromarray(arr, mode="RGB")
    raw = _make_pil_bytes(pil_img, fmt="JPEG", exif_orientation=6)

    res = load_and_validate_image(raw)
    assert res.is_valid is True
    assert res.exif_transposed is True
    # After 90-degree rotation, width=200 and height=420
    assert res.width == 200
    assert res.height == 420


def test_empty_and_corrupted_inputs_fail_gracefully() -> None:
    """Verify empty bytes, random garbage bytes, and missing files do not crash."""
    res_empty = load_and_validate_image(b"")
    assert res_empty.is_valid is False
    assert res_empty.error_message is not None

    res_corrupt = load_and_validate_image(b"not_an_image_at_all_1234567890")
    assert res_corrupt.is_valid is False
    assert res_corrupt.error_message is not None

    res_missing = load_and_validate_image("non_existent_currency_note_file.png")
    assert res_missing.is_valid is False
    assert res_missing.error_message is not None


def test_too_small_and_blank_images_rejected() -> None:
    """Verify images below minimum dimensions or completely uniform are rejected."""
    rng = np.random.default_rng(13)
    tiny = rng.integers(0, 255, size=(40, 80, 3), dtype=np.uint8)
    res_tiny = load_and_validate_image(tiny)
    assert res_tiny.is_valid is False
    assert "too small" in (res_tiny.error_message or "").lower()

    blank = np.zeros((250, 500, 3), dtype=np.uint8)
    res_blank = load_and_validate_image(blank)
    assert res_blank.is_valid is False
    assert "no visual content" in (res_blank.error_message or "").lower()


def test_file_size_limit_enforcement() -> None:
    """Verify files exceeding configured max_file_size_mb are rejected."""
    rng = np.random.default_rng(14)
    arr = rng.integers(0, 255, size=(300, 600, 3), dtype=np.uint8)
    raw = _make_pil_bytes(Image.fromarray(arr), fmt="PNG")
    res = load_and_validate_image(raw, max_file_size_mb=0.001)
    assert res.is_valid is False
    assert "exceeds" in (res.error_message or "").lower()
