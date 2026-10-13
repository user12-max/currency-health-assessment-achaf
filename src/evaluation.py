"""Module 21: Dataset Evaluation Utilities and Synthetic Test Pattern Generator.

Provides quantitative pixel-level localization evaluation metrics (Precision,
Recall, F1-Score, Intersection over Union [IoU], and processing time) when
user-annotated ground-truth masks are supplied.

Also provides a deterministic generator for synthetic currency-like test
patterns (`SYNTHETIC TEST SPECIMEN`) with exact ground-truth defect masks so
that image-processing algorithms can be unit-tested and demonstrated without
treating synthetic data as evidence of real-world banking accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class LocalizationEvaluationMetrics:
    """Pixel-level binary evaluation metrics comparing predicted vs ground-truth masks."""

    precision: float
    recall: float
    f1_score: float
    iou: float
    true_positive_px: int
    false_positive_px: int
    false_negative_px: int
    true_negative_px: int
    processing_time_ms: float

    def to_dict(self) -> Dict[str, float]:
        """Return JSON-serializable dictionary of evaluation metrics."""
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "iou": round(self.iou, 4),
            "true_positive_px": self.true_positive_px,
            "false_positive_px": self.false_positive_px,
            "false_negative_px": self.false_negative_px,
            "true_negative_px": self.true_negative_px,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


def evaluate_mask_against_ground_truth(
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    processing_time_ms: float = 0.0,
) -> LocalizationEvaluationMetrics:
    """Compute Precision, Recall, F1-score, and IoU between two binary masks.

    Only called when an explicit ground-truth mask is supplied. Handles empty
    masks (all-zero) safely without division by zero.
    """
    if predicted_mask is None or ground_truth_mask is None:
        raise ValueError("Both predicted_mask and ground_truth_mask must be provided.")
    if predicted_mask.size == 0 or ground_truth_mask.size == 0:
        raise ValueError("Masks cannot be empty.")

    if predicted_mask.shape[:2] != ground_truth_mask.shape[:2]:
        gh, gw = ground_truth_mask.shape[:2]
        pred_resized = cv2.resize(predicted_mask, (gw, gh), interpolation=cv2.INTER_NEAREST)
    else:
        pred_resized = predicted_mask

    pred_b = pred_resized > 0
    gt_b = ground_truth_mask > 0

    tp = int(np.sum(pred_b & gt_b))
    fp = int(np.sum(pred_b & (~gt_b)))
    fn = int(np.sum((~pred_b) & gt_b))
    tn = int(np.sum((~pred_b) & (~gt_b)))

    if tp == 0 and fp == 0 and fn == 0:
        # Both masks are completely empty (perfect agreement on clean note)
        precision = 1.0
        recall = 1.0
        f1 = 1.0
        iou = 1.0
    else:
        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            (2.0 * precision * recall) / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )
        union = tp + fp + fn
        iou = float(tp) / float(union) if union > 0 else 0.0

    return LocalizationEvaluationMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1,
        iou=iou,
        true_positive_px=tp,
        false_positive_px=fp,
        false_negative_px=fn,
        true_negative_px=tn,
        processing_time_ms=float(processing_time_ms),
    )


def generate_synthetic_test_note(
    variant: str = "clean",
    canvas_width: int = 920,
    canvas_height: int = 460,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Create a synthetic currency-like test image and ground-truth defect masks.

    Clearly labeled on-image as 'SYNTHETIC DIP TEST SPECIMEN' so it is never
    confused with real currency or real-world benchmark data.

    Parameters
    ----------
    variant : str
        - 'clean': Clean rectangular specimen on dark background.
        - 'damaged': Specimen with missing corner, edge nick, vertical crease, and dark stain.
        - 'angled_damaged': Specimen rotated/skewed in perspective with defects.
    """
    rng = np.random.default_rng(seed)
    h, w = canvas_height, canvas_width

    # Dark wood/slate background (RGB)
    bg = np.full((h, w, 3), (28, 32, 38), dtype=np.uint8)

    # Note dimensions inside canvas
    nx1, ny1 = int(w * 0.08), int(h * 0.10)
    nx2, ny2 = int(w * 0.92), int(h * 0.90)
    nw, nh = nx2 - nx1, ny2 - ny1

    # Base note paper color (warm pastel cream/mint tone)
    note = np.full((nh, nw, 3), (222, 232, 214), dtype=np.uint8)

    # Add subtle paper fiber noise
    noise = rng.normal(0, 2.2, size=(nh, nw, 3))
    note = np.clip(note.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Draw decorative inner guilloche-like border and geometric motifs (rich ORB keypoints)
    cv2.rectangle(note, (18, 18), (nw - 19, nh - 19), (75, 115, 90), 2)
    cv2.rectangle(note, (26, 26), (nw - 27, nh - 27), (110, 145, 120), 1)

    # Rosette / lattice pattern on left and center for realistic ORB feature texture
    for cx, cy, rad in [(int(nw * 0.25), int(nh * 0.52), 62), (int(nw * 0.52), int(nh * 0.50), 48)]:
        for r_step in range(12, rad, 10):
            cv2.circle(note, (cx, cy), r_step, (90, 130, 105), 1, cv2.LINE_AA)
        for angle in range(0, 360, 30):
            rad_a = np.radians(angle)
            ex = int(cx + rad * np.cos(rad_a))
            ey = int(cy + rad * np.sin(rad_a))
            cv2.line(note, (cx, cy), (ex, ey), (115, 150, 125), 1, cv2.LINE_AA)

    # Security thread stripe
    tx = int(nw * 0.62)
    cv2.rectangle(note, (tx, 20), (tx + 8, nh - 20), (95, 140, 130), -1)

    # Specimen text
    cv2.putText(
        note,
        "ACHAF SYNTHETIC DIP TEST SPECIMEN",
        (42, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (55, 85, 68),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        note,
        "100 SPECIMEN",
        (int(nw * 0.68), int(nh * 0.28)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        (60, 90, 75),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        note,
        "NOT LEGAL TENDER - FOR ALGORITHM VERIFICATION",
        (42, nh - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 110, 92),
        1,
        cv2.LINE_AA,
    )

    gt_masks: Dict[str, np.ndarray] = {
        "edge_wear": np.zeros((nh, nw), dtype=np.uint8),
        "corner_damage": np.zeros((nh, nw), dtype=np.uint8),
        "fold_lines": np.zeros((nh, nw), dtype=np.uint8),
        "stain_regions": np.zeros((nh, nw), dtype=np.uint8),
        "texture_irregularity": np.zeros((nh, nw), dtype=np.uint8),
    }

    if variant in ("damaged", "angled_damaged"):
        # 1. Missing top-right corner triangle
        c_pts = np.array([[nw - 78, 0], [nw, 0], [nw, 72]], dtype=np.int32)
        cv2.fillPoly(note, [c_pts], (28, 32, 38))
        cv2.fillPoly(gt_masks["corner_damage"], [c_pts], 255)

        # 2. Edge wear / tear notch along top edge
        e_pts = np.array(
            [[int(nw * 0.42), 0], [int(nw * 0.50), 0], [int(nw * 0.46), 28]],
            dtype=np.int32,
        )
        cv2.fillPoly(note, [e_pts], (28, 32, 38))
        cv2.fillPoly(gt_masks["edge_wear"], [e_pts], 255)

        # 3. Prominent diagonal crease/fold shadow + ridge line across interior
        fx1, fy1 = int(nw * 0.36), 24
        fx2, fy2 = int(nw * 0.41), nh - 24
        cv2.line(note, (fx1, fy1), (fx2, fy2), (125, 132, 120), 4, cv2.LINE_AA)
        cv2.line(note, (fx1 + 4, fy1), (fx2 + 4, fy2), (244, 248, 240), 2, cv2.LINE_AA)
        cv2.line(gt_masks["fold_lines"], (fx1, fy1), (fx2, fy2), 255, 8)

        # 4. Localized coffee/ink stain blotch in lower-right interior
        sx, sy = int(nw * 0.76), int(nh * 0.65)
        stain_layer = np.zeros((nh, nw), dtype=np.float32)
        cv2.circle(stain_layer, (sx, sy), 34, 1.0, -1)
        cv2.circle(stain_layer, (sx + 18, sy - 10), 22, 0.85, -1)
        stain_blur = cv2.GaussianBlur(stain_layer, (21, 21), 0)[:, :, np.newaxis]
        stain_color = np.array([135.0, 92.0, 48.0], dtype=np.float32)
        note = np.clip(
            note.astype(np.float32) * (1.0 - 0.68 * stain_blur) + stain_color * (0.68 * stain_blur),
            0,
            255,
        ).astype(np.uint8)
        gt_masks["stain_regions"] = np.where(stain_blur[:, :, 0] > 0.25, 255, 0).astype(np.uint8)

        # 5. Localized surface crumple/roughness patch in lower-left
        tx1, ty1, tx2, ty2 = int(nw * 0.12), int(nh * 0.62), int(nw * 0.24), int(nh * 0.84)
        crumple_noise = rng.normal(0, 26.0, size=(ty2 - ty1, tx2 - tx1, 1))
        note[ty1:ty2, tx1:tx2] = np.clip(
            note[ty1:ty2, tx1:tx2].astype(np.float32) + crumple_noise, 0, 255
        ).astype(np.uint8)
        gt_masks["texture_irregularity"][ty1:ty2, tx1:tx2] = 255

    bg[ny1:ny2, nx1:nx2] = note

    if variant == "angled_damaged":
        # Apply a slight perspective tilt (5 degrees rotation + mild keystone)
        src_q = np.array(
            [[nx1, ny1], [nx2, ny1], [nx2, ny2], [nx1, ny2]], dtype=np.float32
        )
        dst_q = np.array(
            [
                [nx1 + 22, ny1 + 18],
                [nx2 - 14, ny1 - 8],
                [nx2 - 28, ny2 - 16],
                [nx1 + 10, ny2 + 12],
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(src_q, dst_q)
        bg = cv2.warpPerspective(
            bg, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(28, 32, 38)
        )

    return bg, gt_masks
