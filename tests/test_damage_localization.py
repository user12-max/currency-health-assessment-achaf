"""Unit tests for Modules 10, 18, and 21: Damage Detection, Spatial Localization, and Evaluation."""

from __future__ import annotations

import numpy as np

from src.evaluation import (
    evaluate_mask_against_ground_truth,
    generate_synthetic_test_note,
)
from src.pipeline import run_achaf_pipeline


def test_damage_localization_dimensions_and_coordinates() -> None:
    """Verify localization overlays match rectified and original dimensions accurately."""
    dmg_img, _ = generate_synthetic_test_note(variant="angled_damaged")
    out = run_achaf_pipeline(dmg_img, denomination="₹100", note_side="Front")

    assert out.is_success is True
    assert out.segmentation_result is not None
    assert out.localization_result is not None
    assert out.preprocessing_result is not None

    rect_shape = out.segmentation_result.rectified_rgb.shape
    orig_shape = out.preprocessing_result.resized_rgb.shape

    loc = out.localization_result
    assert loc.annotated_rectified_rgb.shape == rect_shape
    assert loc.combined_damage_map_rgb.shape == rect_shape
    assert loc.annotated_original_rgb.shape == orig_shape
    assert loc.has_detected_damage is True
    assert len(loc.localized_regions) >= 1

    # Verify all 4 category masks have exact (H_r, W_r) dimensions
    for key in ("edge_corner", "fold_line", "stain_color", "texture"):
        assert key in loc.category_masks
        assert loc.category_masks[key].shape == rect_shape[:2]

    # Verify all region bounding boxes lie strictly within rectified image bounds
    hr, wr = rect_shape[:2]
    for reg in loc.localized_regions:
        x, y, bw, bh = reg.bbox
        assert 0 <= x < wr
        assert 0 <= y < hr
        assert x + bw <= wr
        assert y + bh <= hr


def test_evaluation_metrics_precision_recall_f1_iou() -> None:
    """Verify Precision, Recall, F1, and IoU calculations on known binary masks."""
    gt = np.zeros((100, 100), dtype=np.uint8)
    pred = np.zeros((100, 100), dtype=np.uint8)

    # Both empty -> perfect agreement (1.0)
    m_empty = evaluate_mask_against_ground_truth(pred, gt, processing_time_ms=12.5)
    assert m_empty.precision == 1.0
    assert m_empty.recall == 1.0
    assert m_empty.f1_score == 1.0
    assert m_empty.iou == 1.0

    # Known partial overlap:
    # GT: 20x20 square (400 px) at [10:30, 10:30]
    # Pred: 20x20 square (400 px) at [10:30, 20:40]
    # Overlap (TP): [10:30, 20:30] = 200 px; FP = 200 px; FN = 200 px; Union = 600 px
    gt[10:30, 10:30] = 255
    pred[10:30, 20:40] = 255
    m = evaluate_mask_against_ground_truth(pred, gt, processing_time_ms=25.0)
    assert np.isclose(m.precision, 0.5)
    assert np.isclose(m.recall, 0.5)
    assert np.isclose(m.f1_score, 0.5)
    assert np.isclose(m.iou, 200.0 / 600.0)
