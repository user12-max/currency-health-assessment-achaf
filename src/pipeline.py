"""End-to-End ACHAF Processing Pipeline Orchestrator.

Executes the complete 12-module pipeline in sequence:
Input image
 -> Module 1: Image validation (`src.image_io`)
 -> Module 2: Capture-quality assessment & preprocessing (`src.image_quality`, `src.preprocessing`)
 -> Module 3: Currency-note segmentation & perspective correction (`src.segmentation`)
 -> Module 4: Normalized region analysis (`src.region_analysis`)
 -> Module 9: Optional reference-assisted alignment & comparison (`src.reference_comparison`)
 -> Module 5: GLCM & LBP texture analysis (`src.texture_analysis`)
 -> Module 6: Canny, Sobel, 4-edge & 4-corner analysis (`src.edge_analysis`)
 -> Module 7: Scale-aware morphological & fold-line analysis (`src.morphology_analysis`)
 -> Module 8: Illumination-normalized HSV/Lab stain & color analysis (`src.color_analysis`)
 -> Module 10: Damage-indicator detection (`src.damage_detection`)
 -> Module 11: Adaptive Multi-Feature Fusion Module (AMFM) (`src.feature_fusion`)
 -> Module 12: Currency Health Index (CHI) calculation (`src.health_scoring`)
 -> Module 18: Spatial damage localization (`src.damage_localization`)
 -> Module 19: Health report generation (`src.report_generation`)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np

from config.settings import AnalysisSettings
from src.color_analysis import ColorAnalysisResult, analyze_color_and_stains
from src.damage_detection import DamageIndicatorResult, run_all_damage_detectors
from src.damage_localization import DamageLocalizationResult, localize_damage_regions
from src.edge_analysis import EdgeAnalysisResult, analyze_edges_and_corners
from src.feature_fusion import AMFMFusionResult, fuse_multi_features
from src.health_scoring import CHIScoringResult, calculate_currency_health_index
from src.image_io import ImageLoadResult, load_and_validate_image
from src.image_quality import CaptureQualityReport
from src.morphology_analysis import MorphologyAnalysisResult, analyze_morphology
from src.preprocessing import PreprocessingResult, preprocess_image
from src.reference_comparison import (
    ReferenceComparisonResult,
    register_and_compare_reference,
)
from src.region_analysis import NoteRegionMap, partition_note_regions
from src.report_generation import build_health_report_dict
from src.segmentation import SegmentationResult, segment_and_rectify_note
from src.texture_analysis import TextureAnalysisResult, analyze_texture


@dataclass
class ACHAFPipelineOutput:
    """Comprehensive container holding all intermediate and final pipeline results."""

    is_success: bool
    error_message: Optional[str] = None
    load_result: Optional[ImageLoadResult] = None
    quality_report: Optional[CaptureQualityReport] = None
    preprocessing_result: Optional[PreprocessingResult] = None
    segmentation_result: Optional[SegmentationResult] = None
    region_map: Optional[NoteRegionMap] = None
    reference_result: Optional[ReferenceComparisonResult] = None
    texture_result: Optional[TextureAnalysisResult] = None
    edge_result: Optional[EdgeAnalysisResult] = None
    morphology_result: Optional[MorphologyAnalysisResult] = None
    color_result: Optional[ColorAnalysisResult] = None
    indicators: Optional[Dict[str, DamageIndicatorResult]] = None
    amfm_result: Optional[AMFMFusionResult] = None
    chi_result: Optional[CHIScoringResult] = None
    localization_result: Optional[DamageLocalizationResult] = None
    report_dict: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0


def run_achaf_pipeline(
    image_source: Union[str, bytes, np.ndarray, Any],
    denomination: str = "Unknown",
    note_side: str = "Unknown",
    analysis_mode: str = "Feature-based analysis",
    reference_source: Optional[Union[str, bytes, np.ndarray, Any]] = None,
    settings: Optional[AnalysisSettings] = None,
) -> ACHAFPipelineOutput:
    """Run the full ACHAF currency health assessment pipeline.

    Parameters
    ----------
    image_source : path | bytes | UploadedFile | np.ndarray
        Primary currency note image to analyze.
    denomination : str
        Selected denomination label (e.g., '₹100', 'Unknown').
    note_side : str
        Selected visible side ('Front', 'Back', or 'Unknown').
    analysis_mode : str
        'Feature-based analysis' or 'Reference-assisted analysis'.
    reference_source : path | bytes | UploadedFile | np.ndarray, optional
        Optional cleaner reference note image of the same denomination and side.
    settings : AnalysisSettings, optional
        Custom analysis parameters and CHI weights.

    Returns
    -------
    ACHAFPipelineOutput
        Complete pipeline artifacts, scores, masks, overlays, and structured report.
    """
    t_start = time.perf_counter()
    cfg = settings or AnalysisSettings()

    # 1. Image acquisition and validation
    load_res = load_and_validate_image(image_source)
    if not load_res.is_valid or load_res.rgb_image is None:
        return ACHAFPipelineOutput(
            is_success=False,
            error_message=load_res.error_message or "Image validation failed.",
            load_result=load_res,
        )

    # 2. Image quality assessment and preprocessing
    pre_res = preprocess_image(load_res.rgb_image, settings=cfg)
    quality_rep = pre_res.quality_report
    for w_msg in load_res.warnings:
        if w_msg not in quality_rep.warnings:
            quality_rep.warnings.insert(0, w_msg)

    # Resize alpha mask if present to match pre_res.resized_rgb
    resized_alpha: Optional[np.ndarray] = None
    if load_res.alpha_mask is not None:
        rh, rw = pre_res.resized_rgb.shape[:2]
        resized_alpha = cv2.resize(
            load_res.alpha_mask, (rw, rh), interpolation=cv2.INTER_NEAREST
        )

    # 3. Currency note segmentation and perspective rectification
    # Run segmentation on denoised_rgb so background-to-note contrast is preserved
    seg_res = segment_and_rectify_note(
        pre_res.denoised_rgb,
        settings=cfg,
        alpha_mask=resized_alpha,
    )

    rect_rgb = seg_res.rectified_rgb
    rect_mask = seg_res.rectified_mask

    # 4. Normalized anatomical region partitioning
    reg_map = partition_note_regions(rect_rgb.shape)

    # 5. Optional Reference-Assisted Alignment & Comparison
    ref_rect_rgb: Optional[np.ndarray] = None
    ref_load_warnings = []
    if (
        analysis_mode == "Reference-assisted analysis"
        and reference_source is not None
    ):
        ref_load = load_and_validate_image(reference_source)
        if ref_load.is_valid and ref_load.rgb_image is not None:
            ref_pre = preprocess_image(ref_load.rgb_image, settings=cfg)
            ref_seg = segment_and_rectify_note(
                ref_pre.denoised_rgb, settings=cfg
            )
            ref_rect_rgb = ref_seg.rectified_rgb
        else:
            ref_load_warnings.append(
                f"Reference image invalid ({ref_load.error_message}); falling back to "
                "single-image feature-based analysis."
            )

    ref_res = register_and_compare_reference(
        test_rectified_rgb=rect_rgb,
        reference_rectified_rgb=ref_rect_rgb,
        test_mask=rect_mask,
        settings=cfg,
    )
    for rw_msg in ref_load_warnings:
        ref_res.warnings.append(rw_msg)
        if not ref_res.status_message:
            ref_res.status_message = rw_msg

    aligned_ref_rgb = ref_res.aligned_reference_rgb if ref_res.alignment_succeeded else None
    aligned_ref_gray = ref_res.aligned_reference_gray if ref_res.alignment_succeeded else None

    # 6. Multi-Feature Extraction (Texture, Edge/Corner, Morphology, Color/Stain)
    tex_res = analyze_texture(
        rectified_rgb=rect_rgb,
        rectified_mask=rect_mask,
        region_map=reg_map,
        reference_gray=aligned_ref_gray,
        sensitivity=cfg.texture_sensitivity,
    )

    edge_res = analyze_edges_and_corners(
        rectified_rgb=rect_rgb,
        rectified_mask=rect_mask,
        region_map=reg_map,
        segmentation_succeeded=seg_res.segmentation_succeeded,
        sensitivity=cfg.edge_sensitivity,
    )

    morph_res = analyze_morphology(
        rectified_rgb=rect_rgb,
        rectified_mask=rect_mask,
        reference_gray=aligned_ref_gray,
        scale_factor=cfg.morphology_scale_factor,
        fold_sensitivity=cfg.fold_sensitivity,
    )

    is_gray_input = load_res.original_mode in ("L", "I", "F", "1")
    color_res = analyze_color_and_stains(
        rectified_rgb=rect_rgb,
        rectified_mask=rect_mask,
        region_map=reg_map,
        reference_rgb=aligned_ref_rgb,
        is_grayscale_input=is_gray_input,
        sensitivity=cfg.stain_sensitivity,
    )

    # 7. Damage-Indicator Detection (6 categories)
    indicators = run_all_damage_detectors(
        edge_result=edge_res,
        morph_result=morph_res,
        texture_result=tex_res,
        color_result=color_res,
        region_map=reg_map,
        quality_report=quality_rep,
        segmentation_succeeded=seg_res.segmentation_succeeded,
        is_grayscale_input=is_gray_input,
    )

    # 8. Adaptive Multi-Feature Fusion Module (AMFM)
    amfm_res = fuse_multi_features(
        edge_result=edge_res,
        morph_result=morph_res,
        texture_result=tex_res,
        color_result=color_res,
        indicators=indicators,
        quality_report=quality_rep,
        segmentation_succeeded=seg_res.segmentation_succeeded,
    )

    # 9. Currency Health Index (CHI) Calculation
    chi_res = calculate_currency_health_index(
        amfm_result=amfm_res,
        weights=cfg.chi_weights,
    )

    # 10. Spatial Damage Localization
    loc_res = localize_damage_regions(
        rectified_rgb=rect_rgb,
        original_resized_rgb=pre_res.resized_rgb,
        indicators=indicators,
        region_map=reg_map,
        inv_homography=seg_res.inv_homography,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # 11. Structured Health Report Generation
    report_dict = build_health_report_dict(
        denomination=denomination,
        note_side=note_side,
        analysis_mode=analysis_mode,
        quality_report=quality_rep,
        segmentation_result=seg_res,
        reference_result=ref_res,
        amfm_result=amfm_res,
        chi_result=chi_res,
        indicators=indicators,
        localization_result=loc_res,
        processing_time_ms=elapsed_ms,
    )

    return ACHAFPipelineOutput(
        is_success=True,
        load_result=load_res,
        quality_report=quality_rep,
        preprocessing_result=pre_res,
        segmentation_result=seg_res,
        region_map=reg_map,
        reference_result=ref_res,
        texture_result=tex_res,
        edge_result=edge_res,
        morphology_result=morph_res,
        color_result=color_res,
        indicators=indicators,
        amfm_result=amfm_res,
        chi_result=chi_res,
        localization_result=loc_res,
        report_dict=report_dict,
        processing_time_ms=elapsed_ms,
    )
