"""Unit tests for Modules 4, 5, 6, 7, 8, and 9: Region, Texture, Edge, Morphology, Color, and Reference Comparison."""

from __future__ import annotations

import numpy as np

from src.color_analysis import analyze_color_and_stains
from src.edge_analysis import analyze_edges_and_corners
from src.evaluation import generate_synthetic_test_note
from src.morphology_analysis import analyze_morphology
from src.reference_comparison import register_and_compare_reference
from src.region_analysis import partition_note_regions
from src.segmentation import segment_and_rectify_note
from src.texture_analysis import analyze_texture, compute_glcm_descriptors


def test_region_partitioning_scales_with_dimensions() -> None:
    """Verify normalized region partitioning covers all 4 edges, 4 corners, center, and patch grid."""
    rmap = partition_note_regions((380, 800, 3))
    assert set(rmap.edge_regions.keys()) == {"top", "bottom", "left", "right"}
    assert set(rmap.corner_regions.keys()) == {
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
    }
    assert rmap.central_region.area > 0
    assert len(rmap.grid_patches) == rmap.grid_rows * rmap.grid_cols
    assert "Corner" in rmap.locate_point_region(5, 5)
    assert "Edge" in rmap.locate_point_region(400, 5)
    assert "Central" in rmap.locate_point_region(400, 190)


def test_glcm_handles_uniform_patch_without_nan() -> None:
    """Verify constant/uniform patches do not produce NaN or Inf in GLCM correlation."""
    uniform_patch = np.full((40, 40), 128, dtype=np.uint8)
    desc = compute_glcm_descriptors(uniform_patch)
    assert all(np.isfinite(v) for v in desc.values())
    assert desc["contrast"] == 0.0
    assert desc["homogeneity"] == 1.0


def test_feature_extractors_distinguish_clean_vs_damaged_notes() -> None:
    """Verify texture, edge/corner, morphology, and color modules score damaged notes worse than clean notes."""
    clean_img, _ = generate_synthetic_test_note(variant="clean")
    dmg_img, _ = generate_synthetic_test_note(variant="damaged")

    seg_clean = segment_and_rectify_note(clean_img)
    seg_dmg = segment_and_rectify_note(dmg_img)

    # Edge & corner analysis
    edge_clean = analyze_edges_and_corners(seg_clean.rectified_rgb, seg_clean.rectified_mask)
    edge_dmg = analyze_edges_and_corners(seg_dmg.rectified_rgb, seg_dmg.rectified_mask)
    assert edge_dmg.overall_edge_wear_score > edge_clean.overall_edge_wear_score
    assert edge_dmg.overall_corner_damage_score > edge_clean.overall_corner_damage_score
    assert edge_dmg.corner_metrics["top_right"].corner_damage_score > 15.0

    # Morphological fold-line analysis
    morph_clean = analyze_morphology(seg_clean.rectified_rgb, seg_clean.rectified_mask)
    morph_dmg = analyze_morphology(seg_dmg.rectified_rgb, seg_dmg.rectified_mask)
    assert morph_dmg.fold_severity_score > morph_clean.fold_severity_score
    assert len(morph_dmg.detected_fold_lines) >= 1

    # Texture analysis
    tex_clean = analyze_texture(seg_clean.rectified_rgb, seg_clean.rectified_mask)
    tex_dmg = analyze_texture(seg_dmg.rectified_rgb, seg_dmg.rectified_mask)
    assert tex_dmg.texture_irregularity_score >= tex_clean.texture_irregularity_score

    # Color & stain analysis
    col_clean = analyze_color_and_stains(seg_clean.rectified_rgb, seg_clean.rectified_mask)
    col_dmg = analyze_color_and_stains(seg_dmg.rectified_rgb, seg_dmg.rectified_mask)
    assert col_dmg.stain_anomaly_score > col_clean.stain_anomaly_score
    assert np.any(col_dmg.stain_mask > 0)


def test_reference_assisted_alignment_success_and_rejection() -> None:
    """Verify ORB+RANSAC succeeds on compatible notes and rejects incompatible images."""
    clean_img, _ = generate_synthetic_test_note(variant="clean")
    dmg_img, _ = generate_synthetic_test_note(variant="damaged")
    seg_clean = segment_and_rectify_note(clean_img)
    seg_dmg = segment_and_rectify_note(dmg_img)

    # Compatible reference note
    ref_ok = register_and_compare_reference(
        test_rectified_rgb=seg_dmg.rectified_rgb,
        reference_rectified_rgb=seg_clean.rectified_rgb,
        test_mask=seg_dmg.rectified_mask,
    )
    assert ref_ok.reference_supplied is True
    assert ref_ok.alignment_succeeded is True
    assert ref_ok.inlier_count >= 18
    assert ref_ok.aligned_reference_rgb is not None
    assert ref_ok.combined_diff_mask is not None

    # Incompatible reference image (random noise pattern with no coherent geometry)
    rng = np.random.default_rng(99)
    incompatible_rgb = rng.integers(0, 255, size=seg_dmg.rectified_rgb.shape, dtype=np.uint8)
    ref_bad = register_and_compare_reference(
        test_rectified_rgb=seg_dmg.rectified_rgb,
        reference_rectified_rgb=incompatible_rgb,
        test_mask=seg_dmg.rectified_mask,
    )
    assert ref_bad.reference_supplied is True
    assert ref_bad.alignment_succeeded is False
    assert ref_bad.rejection_reason is not None
