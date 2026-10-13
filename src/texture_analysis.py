"""Module 5: Texture Analysis.

Implements traditional texture descriptors using Gray Level Co-occurrence Matrix
(GLCM) and Local Binary Patterns (LBP). Performs global and patch-based texture
characterization across the currency note.

Important domain principle:
Currency notes naturally exhibit heterogeneous textures (intaglio printing,
portraits, numerals, and watermark blank areas). Therefore, raw inter-patch
texture variation is NOT automatically treated as physical wear. In single-image
mode, only extreme micro-roughness anomalies (e.g., dense crumple/scuff patterns)
are flagged as potential texture irregularities; when a reliably aligned reference
note is supplied, direct region-by-region GLCM and LBP comparisons are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from src.region_analysis import NoteRegionMap, partition_note_regions


@dataclass
class PatchTextureMetrics:
    """GLCM and LBP metrics for a single spatial patch."""

    patch_name: str
    row: int
    col: int
    glcm_contrast: float
    glcm_homogeneity: float
    glcm_energy: float
    glcm_correlation: float
    lbp_entropy: float
    micro_roughness: float
    anomaly_score: float  # [0.0, 1.0]


@dataclass
class TextureAnalysisResult:
    """Structured output of GLCM and LBP texture analysis."""

    global_glcm_contrast: float
    global_glcm_homogeneity: float
    global_glcm_energy: float
    global_glcm_correlation: float
    global_lbp_entropy: float
    mean_micro_roughness: float
    patch_metrics: List[PatchTextureMetrics]
    texture_irregularity_score: float  # [0, 100], 0 = clean, 100 = severe irregularity
    texture_condition_score: float     # [0, 100], 100 = best condition
    anomaly_mask: np.ndarray           # Binary uint8 mask (255 = localized texture anomaly)
    heatmap_gray: np.ndarray           # Continuous uint8 map [0..255] of patch anomaly
    lbp_image: np.ndarray              # Normalized uint8 visualization of LBP codes
    used_reference: bool = False
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_summary_dict(self) -> Dict[str, float]:
        """Return key numerical texture descriptors."""
        return {
            "global_glcm_contrast": round(self.global_glcm_contrast, 4),
            "global_glcm_homogeneity": round(self.global_glcm_homogeneity, 4),
            "global_glcm_energy": round(self.global_glcm_energy, 4),
            "global_glcm_correlation": round(self.global_glcm_correlation, 4),
            "global_lbp_entropy": round(self.global_lbp_entropy, 4),
            "mean_micro_roughness": round(self.mean_micro_roughness, 4),
            "texture_irregularity_score": round(self.texture_irregularity_score, 2),
            "texture_condition_score": round(self.texture_condition_score, 2),
        }


def compute_glcm_descriptors(gray_patch: np.ndarray, levels: int = 32) -> Dict[str, float]:
    """Compute GLCM contrast, homogeneity, energy, and correlation safely.

    Quantizes intensities to `levels` gray levels for computational speed and
    statistical stability on small patches. Handles uniform patches without NaN.
    """
    if gray_patch is None or gray_patch.size < 4:
        return {
            "contrast": 0.0,
            "homogeneity": 1.0,
            "energy": 1.0,
            "correlation": 1.0,
        }

    # Quantize 0..255 -> 0..(levels-1)
    quantized = np.clip((gray_patch.astype(np.int32) * levels) // 256, 0, levels - 1).astype(
        np.uint8
    )

    if float(np.std(quantized)) < 1e-4:
        # Completely uniform patch: correlation is undefined in standard formula (0/0),
        # so return 1.0 safely.
        return {
            "contrast": 0.0,
            "homogeneity": 1.0,
            "energy": 1.0,
            "correlation": 1.0,
        }

    angles = [0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    contrast = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    energy = float(np.mean(graycoprops(glcm, "energy")))
    corr_arr = graycoprops(glcm, "correlation")
    corr_finite = corr_arr[np.isfinite(corr_arr)]
    correlation = float(np.mean(corr_finite)) if corr_finite.size > 0 else 1.0

    return {
        "contrast": contrast if np.isfinite(contrast) else 0.0,
        "homogeneity": homogeneity if np.isfinite(homogeneity) else 1.0,
        "energy": energy if np.isfinite(energy) else 1.0,
        "correlation": float(np.clip(correlation, -1.0, 1.0)),
    }


def compute_lbp_features(
    gray_image: np.ndarray, n_points: int = 8, radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute uniform Local Binary Pattern (LBP) image, normalized histogram, and entropy."""
    lbp = local_binary_pattern(gray_image, P=n_points, R=radius, method="uniform")
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=False)
    hist_f = hist.astype(np.float64)
    total = max(float(np.sum(hist_f)), 1.0)
    probs = hist_f / total
    nonzero = probs[probs > 1e-12]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))

    # Convert LBP codes to 0..255 uint8 for visualization
    lbp_vis = np.clip((lbp / float(max(1, n_bins - 1))) * 255.0, 0, 255).astype(np.uint8)
    return lbp, probs, entropy, lbp_vis


def _compute_micro_roughness_map(gray: np.ndarray) -> np.ndarray:
    """Measure fine-scale non-directional surface roughness (crumple / abrasion residual).

    Clean printed graphics have strong directional edges (text, Guilloche lines),
    whereas physical surface crumpling and scuffing introduce high-frequency
    multi-directional residual variations relative to a structure-preserving filter.
    """
    gray_f = gray.astype(np.float32)
    # Smooth with a small median filter to remove 1-pixel dust while keeping print edges
    med = cv2.medianBlur(gray, 3).astype(np.float32)
    # Bilateral baseline captures clean step edges of printed artwork
    bilat = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=35).astype(np.float32)
    residual = np.abs(med - bilat)
    return residual


def analyze_texture(
    rectified_rgb: np.ndarray,
    rectified_mask: Optional[np.ndarray] = None,
    region_map: Optional[NoteRegionMap] = None,
    reference_gray: Optional[np.ndarray] = None,
    sensitivity: float = 1.0,
) -> TextureAnalysisResult:
    """Run GLCM and LBP texture analysis across the rectified currency note.

    Parameters
    ----------
    rectified_rgb : np.ndarray
        Rectified RGB uint8 image of the currency note.
    rectified_mask : np.ndarray, optional
        Foreground binary mask (255 = valid note pixel).
    region_map : NoteRegionMap, optional
        Precomputed anatomical region partition.
    reference_gray : np.ndarray, optional
        Aligned grayscale reference image of the same shape, when available.
    sensitivity : float
        Sensitivity multiplier in [0.5, 2.0].
    """
    if rectified_rgb is None or rectified_rgb.size == 0:
        raise ValueError("Cannot run texture analysis on an empty image.")

    gray = (
        rectified_rgb
        if rectified_rgb.ndim == 2
        else cv2.cvtColor(rectified_rgb, cv2.COLOR_RGB2GRAY)
    )
    h, w = gray.shape[:2]
    rmap = region_map or partition_note_regions((h, w))
    sens = float(np.clip(sensitivity, 0.5, 2.0))

    # Global GLCM & LBP descriptors
    global_glcm = compute_glcm_descriptors(gray)
    lbp_codes, _, global_lbp_entropy, lbp_vis = compute_lbp_features(gray)
    roughness_map = _compute_micro_roughness_map(gray)

    used_ref = (
        reference_gray is not None
        and isinstance(reference_gray, np.ndarray)
        and reference_gray.shape[:2] == (h, w)
    )
    if used_ref:
        ref_lbp_codes, _, _, _ = compute_lbp_features(reference_gray)
        ref_roughness_map = _compute_micro_roughness_map(reference_gray)
    else:
        ref_lbp_codes = None
        ref_roughness_map = None

    patch_metrics: List[PatchTextureMetrics] = []
    n_bins = 10

    # First pass: collect raw descriptors per patch
    raw_roughness_list: List[float] = []
    raw_homog_list: List[float] = []

    for idx, pbox in enumerate(rmap.grid_patches):
        r = idx // rmap.grid_cols
        c = idx % rmap.grid_cols
        patch_gray = pbox.extract_slice(gray)
        patch_lbp = pbox.extract_slice(lbp_codes)
        patch_rough = pbox.extract_slice(roughness_map)

        glcm_d = compute_glcm_descriptors(patch_gray)
        hist, _ = np.histogram(patch_lbp.ravel(), bins=n_bins, range=(0, n_bins))
        probs = hist.astype(np.float64) / max(float(np.sum(hist)), 1.0)
        nz = probs[probs > 1e-12]
        p_entropy = float(-np.sum(nz * np.log2(nz)))
        p_rough = float(np.mean(patch_rough))

        raw_roughness_list.append(p_rough)
        raw_homog_list.append(glcm_d["homogeneity"])

        patch_metrics.append(
            PatchTextureMetrics(
                patch_name=pbox.name,
                row=r,
                col=c,
                glcm_contrast=glcm_d["contrast"],
                glcm_homogeneity=glcm_d["homogeneity"],
                glcm_energy=glcm_d["energy"],
                glcm_correlation=glcm_d["correlation"],
                lbp_entropy=p_entropy,
                micro_roughness=p_rough,
                anomaly_score=0.0,
            )
        )

    median_roughness = float(np.median(raw_roughness_list)) if raw_roughness_list else 0.0
    iqr_roughness = float(
        np.percentile(raw_roughness_list, 75) - np.percentile(raw_roughness_list, 25)
    ) if len(raw_roughness_list) >= 4 else 1.0
    scale_rough = max(iqr_roughness, 2.2)

    heatmap_grid = np.zeros((rmap.grid_rows, rmap.grid_cols), dtype=np.float32)

    # Second pass: compute anomaly score per patch
    for idx, pm in enumerate(patch_metrics):
        pbox = rmap.grid_patches[idx]

        if used_ref and reference_gray is not None and ref_lbp_codes is not None and ref_roughness_map is not None:
            ref_patch_gray = pbox.extract_slice(reference_gray)
            ref_glcm = compute_glcm_descriptors(ref_patch_gray)
            ref_patch_lbp = pbox.extract_slice(ref_lbp_codes)
            ref_hist, _ = np.histogram(ref_patch_lbp.ravel(), bins=n_bins, range=(0, n_bins))
            ref_probs = ref_hist.astype(np.float64) / max(float(np.sum(ref_hist)), 1.0)

            test_patch_lbp = pbox.extract_slice(lbp_codes)
            test_hist, _ = np.histogram(test_patch_lbp.ravel(), bins=n_bins, range=(0, n_bins))
            test_probs = test_hist.astype(np.float64) / max(float(np.sum(test_hist)), 1.0)

            lbp_l1 = float(0.5 * np.sum(np.abs(test_probs - ref_probs)))
            homog_diff = abs(pm.glcm_homogeneity - ref_glcm["homogeneity"])
            ref_rough = float(np.mean(pbox.extract_slice(ref_roughness_map)))
            rough_excess = max(0.0, pm.micro_roughness - ref_rough) / 6.0

            raw_score = (0.45 * (lbp_l1 / 0.28) + 0.30 * (homog_diff / 0.22) + 0.25 * rough_excess) * sens
            pm.anomaly_score = float(np.clip(raw_score, 0.0, 1.0))
        else:
            # Single-image mode: avoid flagging normal printed vs blank areas!
            # Only flag patches where micro-roughness (crumpling/abrasion noise) is both
            # high in absolute terms (> 4.8) AND strongly elevated above the note median,
            # accompanied by low GLCM homogeneity.
            excess_rough = max(0.0, pm.micro_roughness - max(median_roughness + 1.8, 4.5))
            z_rough = excess_rough / scale_rough
            homog_penalty = max(0.0, 0.55 - pm.glcm_homogeneity)
            raw_score = (0.70 * (z_rough / 2.5) + 0.30 * (homog_penalty / 0.30)) * sens
            if pm.micro_roughness < 4.2:
                raw_score *= 0.25
            pm.anomaly_score = float(np.clip(raw_score, 0.0, 1.0))

        heatmap_grid[pm.row, pm.col] = pm.anomaly_score

    # Upsample patch heatmap smoothly to full rectified note size
    heatmap_float = cv2.resize(heatmap_grid, (w, h), interpolation=cv2.INTER_LINEAR)
    # Modulate by pixel-level micro-roughness so localized scuffs have sharper boundaries
    norm_pixel_rough = np.clip((roughness_map - 2.5) / 12.0, 0.2, 1.2)
    combined_field = np.clip(heatmap_float * norm_pixel_rough, 0.0, 1.0)

    if rectified_mask is not None:
        valid_mask = rectified_mask > 0
        combined_field = np.where(valid_mask, combined_field, 0.0)

    heatmap_gray = np.clip(combined_field * 255.0, 0, 255).astype(np.uint8)

    # Binary anomaly mask for patches/pixels exceeding anomaly threshold
    thresh_val = int(round(np.clip(135.0 / sens, 85.0, 185.0)))
    _, anomaly_mask = cv2.threshold(heatmap_gray, thresh_val, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    anomaly_mask = cv2.morphologyEx(anomaly_mask, cv2.MORPH_OPEN, kernel)

    mean_roughness = float(np.mean(raw_roughness_list)) if raw_roughness_list else 0.0
    all_patch_scores = [p.anomaly_score for p in patch_metrics]
    top_k_mean = (
        float(np.mean(sorted(all_patch_scores, reverse=True)[: max(1, len(all_patch_scores) // 4)]))
        if all_patch_scores
        else 0.0
    )
    overall_mean = float(np.mean(all_patch_scores)) if all_patch_scores else 0.0

    # Combine top-quartile patch anomaly and overall mean anomaly into [0, 100]
    irregularity_score = float(
        np.clip((0.60 * top_k_mean + 0.40 * overall_mean) * 100.0, 0.0, 100.0)
    )
    condition_score = float(np.clip(100.0 - irregularity_score, 0.0, 100.0))

    if used_ref:
        explanation = (
            f"Region-wise GLCM and LBP comparison against aligned reference note yielded "
            f"texture condition score {condition_score:.1f}/100 "
            f"(mean GLCM homogeneity={global_glcm['homogeneity']:.2f}, LBP entropy={global_lbp_entropy:.2f})."
        )
    else:
        explanation = (
            f"Patch-based GLCM and uniform LBP analysis yielded texture condition score "
            f"{condition_score:.1f}/100 (global GLCM homogeneity={global_glcm['homogeneity']:.2f}, "
            f"contrast={global_glcm['contrast']:.2f}, LBP entropy={global_lbp_entropy:.2f}). "
            "Natural print-pattern variations were separated from isotropic surface roughness."
        )

    return TextureAnalysisResult(
        global_glcm_contrast=global_glcm["contrast"],
        global_glcm_homogeneity=global_glcm["homogeneity"],
        global_glcm_energy=global_glcm["energy"],
        global_glcm_correlation=global_glcm["correlation"],
        global_lbp_entropy=global_lbp_entropy,
        mean_micro_roughness=mean_roughness,
        patch_metrics=patch_metrics,
        texture_irregularity_score=irregularity_score,
        texture_condition_score=condition_score,
        anomaly_mask=anomaly_mask,
        heatmap_gray=heatmap_gray,
        lbp_image=lbp_vis,
        used_reference=used_ref,
        explanation=explanation,
    )
