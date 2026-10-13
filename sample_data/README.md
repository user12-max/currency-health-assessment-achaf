# Sample Data Directory (`sample_data/`)

## Purpose and Academic Disclaimer

This directory contains **synthetic test specimen images** generated programmatically by `src/evaluation.py` (`generate_synthetic_test_note`) for verifying and demonstrating the **ACHAF (Adaptive Currency Health Assessment Framework)** Digital Image Processing pipeline:

1. `synthetic_specimen_clean.png` — Intact rectangular synthetic test specimen on a contrasting dark background (reference/clean baseline).
2. `synthetic_specimen_damaged.png` — Synthetic specimen containing controlled, known defects:
   - Missing top-right corner triangle.
   - Top-edge tear notch.
   - Elongated interior crease/fold shadow-ridge line.
   - Lower-right chromatic/dark stain blotch.
   - Lower-left high-frequency surface crumple/roughness patch.
3. `synthetic_specimen_angled_damaged.png` — Perspective-tilted and rotated version of the damaged specimen for verifying quadrilateral segmentation and 4-point perspective homography rectification.
4. `synthetic_specimen_damaged_gt_mask.png` — Combined binary ground-truth defect mask corresponding to the rectified damaged specimen, used to test Precision, Recall, F1-score, and Intersection over Union (IoU) in `src/evaluation.py`.

> **Important Distinction:** These synthetic test patterns are strictly for deterministic unit testing and algorithm verification. They are **NOT** photographs of real legal tender and must **NOT** be cited as evidence of real-world banking accuracy.

## Using Your Own Real Currency Note Photographs

To evaluate genuine currency notes:
1. Launch the Streamlit application (`python -m streamlit run app.py`).
2. In the sidebar under **1. Note Image Input**, select **Upload Currency Image**.
3. Upload a photograph (`.png`, `.jpg`, `.jpeg`, or `.webp`, up to 15 MB) of a genuine currency note placed flat against a contrasting, uncluttered background under even diffuse lighting.
4. Optionally upload a photograph of a cleaner note of the **same denomination, side, and design** when using **Reference-assisted analysis**.
