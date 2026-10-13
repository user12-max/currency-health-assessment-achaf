"""Capture-Quality Assessment Module.

Evaluates photographic capture quality (brightness, contrast, sharpness/blur via
Laplacian variance, exposure clipping, illumination uniformity, and resolution)
separately from physical currency-note condition.

A blurry or poorly lit photograph does NOT mean the physical note is damaged;
instead, poor capture quality triggers warnings and marks the assessment as
provisional.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np

from config.settings import (
    BLUR_LAPLACIAN_SEVERE_THRESHOLD,
    BLUR_LAPLACIAN_WARN_THRESHOLD,
    BRIGHTNESS_HIGH_THRESHOLD,
    BRIGHTNESS_LOW_THRESHOLD,
    CONTRAST_LOW_STD_THRESHOLD,
    OVEREXPOSED_FRACTION_THRESHOLD,
    RECOMMENDED_MIN_HEIGHT,
    RECOMMENDED_MIN_WIDTH,
    UNDEREXPOSED_FRACTION_THRESHOLD,
)


@dataclass
class CaptureQualityReport:
    """Structured capture-quality indicators for an input image."""

    width: int
    height: int
    mean_brightness: float
    std_contrast: float
    dynamic_range_p95_p5: float
    blur_laplacian_variance: float
    overexposed_fraction: float
    underexposed_fraction: float
    illumination_nonuniformity: float
    is_blurry: bool
    is_severely_blurry: bool
    is_overexposed: bool
    is_underexposed: bool
    is_low_contrast: bool
    is_low_resolution: bool
    is_provisional: bool
    quality_rating: str  # "Good", "Acceptable", or "Poor (Provisional)"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable dictionary of capture quality metrics."""
        return {
            "width": self.width,
            "height": self.height,
            "mean_brightness": round(self.mean_brightness, 2),
            "std_contrast": round(self.std_contrast, 2),
            "dynamic_range_p95_p5": round(self.dynamic_range_p95_p5, 2),
            "blur_laplacian_variance": round(self.blur_laplacian_variance, 2),
            "overexposed_fraction": round(self.overexposed_fraction, 4),
            "underexposed_fraction": round(self.underexposed_fraction, 4),
            "illumination_nonuniformity": round(self.illumination_nonuniformity, 4),
            "is_blurry": self.is_blurry,
            "is_severely_blurry": self.is_severely_blurry,
            "is_overexposed": self.is_overexposed,
            "is_underexposed": self.is_underexposed,
            "is_low_contrast": self.is_low_contrast,
            "is_low_resolution": self.is_low_resolution,
            "is_provisional": self.is_provisional,
            "quality_rating": self.quality_rating,
            "warnings": list(self.warnings),
        }


def assess_capture_quality(rgb_image: np.ndarray) -> CaptureQualityReport:
    """Assess photographic capture quality of an RGB uint8 image.

    Parameters
    ----------
    rgb_image : np.ndarray
        Input RGB image of shape (H, W, 3) with dtype uint8.

    Returns
    -------
    CaptureQualityReport
        Comprehensive quality metrics, flags, and actionable warnings.
    """
    if rgb_image is None or rgb_image.size == 0:
        raise ValueError("Cannot assess quality of an empty image.")

    if rgb_image.ndim == 2:
        gray = rgb_image.astype(np.uint8)
    else:
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape[:2]
    gray_f = gray.astype(np.float64)

    mean_brightness = float(np.mean(gray_f))
    std_contrast = float(np.std(gray_f))
    p5, p95 = np.percentile(gray_f, [5.0, 95.0])
    dynamic_range = float(p95 - p5)

    # Standardize image scale for Laplacian variance so blur metric is consistent
    # across different upload resolutions
    max_dim = max(width, height)
    if max_dim > 800:
        scale = 800.0 / float(max_dim)
        eval_gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        eval_gray = gray

    laplacian = cv2.Laplacian(eval_gray, cv2.CV_64F)
    blur_var = float(np.var(laplacian))

    overexposed_fraction = float(np.mean(gray >= 250))
    underexposed_fraction = float(np.mean(gray <= 10))

    # Measure coarse spatial illumination gradient (4x4 grid of local means)
    grid_h = max(1, height // 4)
    grid_w = max(1, width // 4)
    local_means: List[float] = []
    for r in range(0, height - grid_h + 1, grid_h):
        for c in range(0, width - grid_w + 1, grid_w):
            patch = gray_f[r : r + grid_h, c : c + grid_w]
            if patch.size > 0:
                local_means.append(float(np.mean(patch)))
    if len(local_means) > 1 and mean_brightness > 1e-3:
        illum_nonuniformity = float(np.std(local_means) / max(mean_brightness, 1.0))
    else:
        illum_nonuniformity = 0.0

    # Evaluate condition flags
    is_blurry = blur_var < BLUR_LAPLACIAN_WARN_THRESHOLD
    is_severely_blurry = blur_var < BLUR_LAPLACIAN_SEVERE_THRESHOLD
    is_overexposed = (
        mean_brightness > BRIGHTNESS_HIGH_THRESHOLD
        or overexposed_fraction > OVEREXPOSED_FRACTION_THRESHOLD
    )
    is_underexposed = (
        mean_brightness < BRIGHTNESS_LOW_THRESHOLD
        or underexposed_fraction > UNDEREXPOSED_FRACTION_THRESHOLD
    )
    is_low_contrast = std_contrast < CONTRAST_LOW_STD_THRESHOLD
    is_low_resolution = (
        max(width, height) < RECOMMENDED_MIN_WIDTH
        or min(width, height) < RECOMMENDED_MIN_HEIGHT
    )

    warnings: List[str] = []
    if is_severely_blurry:
        warnings.append(
            f"Image appears severely blurry (Laplacian variance={blur_var:.1f} < "
            f"{BLUR_LAPLACIAN_SEVERE_THRESHOLD:.0f}). Fine creases and surface texture "
            "cannot be assessed with high confidence."
        )
    elif is_blurry:
        warnings.append(
            f"Moderate image blur detected (Laplacian variance={blur_var:.1f}). "
            "Subtle fold lines and micro-texture features may be attenuated."
        )

    if is_overexposed:
        warnings.append(
            f"High brightness / overexposure detected (mean={mean_brightness:.1f}, "
            f"clipped bright pixels={overexposed_fraction * 100:.1f}%). Fading and "
            "light stain indicators may be less reliable."
        )
    if is_underexposed:
        warnings.append(
            f"Low brightness / underexposure detected (mean={mean_brightness:.1f}, "
            f"clipped dark pixels={underexposed_fraction * 100:.1f}%). Shadow regions "
            "may limit color and stain analysis."
        )
    if is_low_contrast:
        warnings.append(
            f"Low global contrast (std={std_contrast:.1f}). CLAHE contrast enhancement "
            "will be applied during preprocessing."
        )
    if illum_nonuniformity > 0.25:
        warnings.append(
            f"Uneven lighting or shadow gradient detected across the frame "
            f"(non-uniformity index={illum_nonuniformity:.2f}). Illumination normalization "
            "is recommended."
        )
    if is_low_resolution:
        warnings.append(
            f"Image resolution ({width}x{height} px) is below the recommended "
            f"{RECOMMENDED_MIN_WIDTH}x{RECOMMENDED_MIN_HEIGHT} px threshold."
        )

    is_provisional = bool(
        is_severely_blurry
        or (is_overexposed and overexposed_fraction > 0.45)
        or (is_underexposed and underexposed_fraction > 0.50)
        or (is_low_resolution and is_blurry)
    )

    if is_provisional:
        quality_rating = "Poor (Provisional)"
        warnings.append(
            "Capture quality is poor; overall health assessment is marked as PROVISIONAL."
        )
    elif warnings:
        quality_rating = "Acceptable"
    else:
        quality_rating = "Good"

    return CaptureQualityReport(
        width=width,
        height=height,
        mean_brightness=mean_brightness,
        std_contrast=std_contrast,
        dynamic_range_p95_p5=dynamic_range,
        blur_laplacian_variance=blur_var,
        overexposed_fraction=overexposed_fraction,
        underexposed_fraction=underexposed_fraction,
        illumination_nonuniformity=illum_nonuniformity,
        is_blurry=is_blurry,
        is_severely_blurry=is_severely_blurry,
        is_overexposed=is_overexposed,
        is_underexposed=is_underexposed,
        is_low_contrast=is_low_contrast,
        is_low_resolution=is_low_resolution,
        is_provisional=is_provisional,
        quality_rating=quality_rating,
        warnings=warnings,
    )
