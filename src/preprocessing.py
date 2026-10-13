"""Module 2: Image Preprocessing.

Implements aspect-ratio-preserving resizing, grayscale conversion, edge-preserving
or gentle denoising (bilateral, Gaussian, or median), CLAHE contrast enhancement,
optional illumination normalization, and conversions to HSV and CIE Lab color spaces.

Care is taken to preserve fine structural details (thin creases, small edge tears)
by avoiding aggressive smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import AnalysisSettings, MAX_PROCESSING_DIMENSION
from src.image_quality import CaptureQualityReport, assess_capture_quality


@dataclass
class PreprocessingResult:
    """Container for all preprocessed image representations and quality metrics."""

    original_rgb: np.ndarray
    resized_rgb: np.ndarray
    denoised_rgb: np.ndarray
    illumination_corrected_rgb: np.ndarray
    enhanced_rgb: np.ndarray
    gray: np.ndarray
    denoised_gray: np.ndarray
    clahe_gray: np.ndarray
    hsv: np.ndarray
    lab: np.ndarray
    scale_x: float
    scale_y: float
    quality_report: CaptureQualityReport


def resize_preserving_aspect(
    image: np.ndarray, max_dimension: int = MAX_PROCESSING_DIMENSION
) -> Tuple[np.ndarray, float, float]:
    """Resize an image while preserving aspect ratio if its largest side exceeds max_dimension.

    Returns
    -------
    resized : np.ndarray
        Resized (or original) image.
    scale_x : float
        Ratio `resized_width / original_width`.
    scale_y : float
        Ratio `resized_height / original_height`.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot resize an empty image.")

    orig_h, orig_w = image.shape[:2]
    max_side = max(orig_w, orig_h)
    if max_side <= max_dimension or max_dimension <= 0:
        return image.copy(), 1.0, 1.0

    scale = float(max_dimension) / float(max_side)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scale_x = float(new_w) / float(orig_w)
    scale_y = float(new_h) / float(orig_h)
    return resized, scale_x, scale_y


def apply_denoising(
    rgb_image: np.ndarray,
    method: str = "bilateral",
    strength: int = 5,
) -> np.ndarray:
    """Apply gentle denoising while preserving edges and thin crease-like lines.

    Parameters
    ----------
    rgb_image : np.ndarray
        Input RGB uint8 image.
    method : str
        One of 'bilateral', 'gaussian', 'median', or 'none'.
    strength : int
        Kernel diameter/size (clamped to a gentle range [3, 9] so thin features
        are not obliterated).
    """
    method_clean = (method or "bilateral").strip().lower()
    if method_clean == "none" or strength <= 1:
        return rgb_image.copy()

    # Clamp kernel size to odd integer in [3, 9] to prevent over-smoothing
    ksize = max(3, min(9, int(strength)))
    if ksize % 2 == 0:
        ksize += 1

    if method_clean == "gaussian":
        return cv2.GaussianBlur(rgb_image, (ksize, ksize), sigmaX=0.8)
    if method_clean == "median":
        return cv2.medianBlur(rgb_image, ksize)

    # Default: bilateral filter preserves sharp edges and fold boundaries
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    filtered_bgr = cv2.bilateralFilter(
        bgr, d=ksize, sigmaColor=25.0, sigmaSpace=25.0
    )
    return cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)


def normalize_illumination(rgb_image: np.ndarray) -> np.ndarray:
    """Suppress broad lighting gradients across the image using Lab L-channel normalization.

    Uses a large-scale Gaussian background estimate on the L channel so slow shadows
    are flattened while localized stains, print patterns, and crease shadows remain intact.
    """
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    h, w = l_chan.shape[:2]
    # Kernel size ~25% of largest image dimension (must be odd and >= 31)
    ksize = max(31, int(round(max(h, w) * 0.25)))
    if ksize % 2 == 0:
        ksize += 1

    l_float = l_chan.astype(np.float32)
    bg_lum = cv2.GaussianBlur(l_float, (ksize, ksize), 0)
    mean_lum = float(np.mean(l_float))
    target_base = np.clip(mean_lum, 115.0, 175.0)

    # Divide by smooth illumination envelope and rescale to target base luminance
    corrected_l = (l_float / np.maximum(bg_lum, 15.0)) * target_base
    # Blend 65% normalized + 35% original L so natural contrast is not over-flattened
    blended_l = 0.65 * corrected_l + 0.35 * l_float
    l_norm = np.clip(blended_l, 0, 255).astype(np.uint8)

    merged_lab = cv2.merge([l_norm, a_chan, b_chan])
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)


def apply_clahe_enhancement(
    rgb_image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to Lab L-channel and grayscale.

    Returns
    -------
    enhanced_rgb : np.ndarray
        RGB image with CLAHE-enhanced luminance.
    clahe_gray : np.ndarray
        Single-channel CLAHE-enhanced grayscale image.
    """
    clip = float(np.clip(clip_limit, 0.5, 6.0))
    grid = max(2, min(16, int(tile_grid_size)))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))

    lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_clahe = clahe.apply(l_chan)
    enhanced_lab = cv2.merge([l_clahe, a_chan, b_chan])
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    clahe_gray = clahe.apply(gray)
    return enhanced_rgb, clahe_gray


def preprocess_image(
    rgb_image: np.ndarray,
    settings: Optional[AnalysisSettings] = None,
) -> PreprocessingResult:
    """Execute the complete Module 2 preprocessing pipeline on an RGB image.

    Parameters
    ----------
    rgb_image : np.ndarray
        Validated RGB uint8 image.
    settings : AnalysisSettings, optional
        Pipeline configuration parameters.

    Returns
    -------
    PreprocessingResult
        All intermediate and final preprocessed representations + capture quality report.
    """
    cfg = settings or AnalysisSettings()

    # 1. Assess capture quality on the input image
    quality_report = assess_capture_quality(rgb_image)

    # 2. Resize preserving aspect ratio
    resized_rgb, scale_x, scale_y = resize_preserving_aspect(
        rgb_image, max_dimension=cfg.target_max_width
    )

    # 3. Gentle denoising (adaptive: if image is already blurry, reduce smoothing)
    effective_method = cfg.denoise_method
    effective_strength = cfg.denoise_strength
    if quality_report.is_blurry and effective_method != "none":
        effective_strength = min(effective_strength, 3)

    denoised_rgb = apply_denoising(
        resized_rgb, method=effective_method, strength=effective_strength
    )

    # 4. Optional illumination normalization
    if cfg.apply_illumination_norm:
        illum_rgb = normalize_illumination(denoised_rgb)
    else:
        illum_rgb = denoised_rgb.copy()

    # 5. CLAHE contrast enhancement
    enhanced_rgb, clahe_gray = apply_clahe_enhancement(
        illum_rgb,
        clip_limit=cfg.clahe_clip_limit,
        tile_grid_size=cfg.clahe_tile_grid_size,
    )

    # 6. Grayscale, HSV, and Lab representations
    gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
    denoised_gray = cv2.cvtColor(illum_rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(illum_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(illum_rgb, cv2.COLOR_RGB2LAB)

    return PreprocessingResult(
        original_rgb=rgb_image,
        resized_rgb=resized_rgb,
        denoised_rgb=denoised_rgb,
        illumination_corrected_rgb=illum_rgb,
        enhanced_rgb=enhanced_rgb,
        gray=gray,
        denoised_gray=denoised_gray,
        clahe_gray=clahe_gray,
        hsv=hsv,
        lab=lab,
        scale_x=scale_x,
        scale_y=scale_y,
        quality_report=quality_report,
    )
