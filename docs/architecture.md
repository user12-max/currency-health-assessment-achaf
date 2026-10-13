# ACHAF System Architecture

## 1. Architectural Overview

**ACHAF (Adaptive Currency Health Assessment Framework)** is a modular, CPU-based Digital Image Processing (DIP) system designed to inspect photographs of currency notes, isolate and rectify the note geometry, extract multi-domain visual features (edge, corner, morphological, texture, and color/stain), fuse those features via the **Adaptive Multi-Feature Fusion Module (AMFM)**, compute the **Currency Health Index (CHI)**, localize damaged regions spatially, and export structured diagnostic reports.

The codebase strictly separates user-interface presentation (`app.py`), central configuration (`config/settings.py`), pipeline orchestration (`src/pipeline.py`), and individual DIP algorithm modules (`src/*.py`).

---

## 2. Information Flow & Pipeline Stages

```
[Input Source: File / Bytes / Stream / NumPy]
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Module 1: Image Acquisition & Validation (src/image_io.py)│
│  • Format verification (PNG, JPG, JPEG, WebP)            │
│  • EXIF orientation correction                           │
│  • RGB/BGR/Grayscale/RGBA transparency normalization     │
│  • File size (<=15 MB) & minimum dimension checks        │
└──────────────────────────┬───────────────────────────────┘
                           │ Validated RGB + Alpha Mask
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Module 2: Capture Quality & Preprocessing                │
│ (src/image_quality.py, src/preprocessing.py)             │
│  • Brightness, contrast, exposure & Laplacian blur check │
│  • Provisional flagging for poor capture quality         │
│  • Aspect-ratio preserving resize                        │
│  • Edge-preserving bilateral / Gaussian / median filter  │
│  • Lab L-channel illumination normalization              │
│  • CLAHE contrast enhancement & HSV / CIE Lab conversion │
└──────────────────────────┬───────────────────────────────┘
                           │ Preprocessed RGB & Quality Report
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Module 3: Currency Note Segmentation & Rectification     │
│ (src/segmentation.py)                                    │
│  • Multi-cue foreground mask (Lab border color model +   │
│    Otsu threshold + Canny edges + morphology)            │
│  • External contour extraction & area ratio validation   │
│  • 4-corner quadrilateral fitting & point ordering       │
│  • Perspective warp (H) & inverse homography (H^-1)      │
│  • Safe fallback to manual crop / full-frame mode        │
└──────────────────────────┬───────────────────────────────┘
                           │ Rectified RGB, Mask, H, H^-1
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Module 4: Normalized Region Analysis                     │
│ (src/region_analysis.py)                                 │
│  • Scale-invariant normalized coordinate partitioning:   │
│    - 4 Boundary Edge strips (top, bottom, left, right)   │
│    - 4 Corner boxes (TL, TR, BL, BR)                     │
│    - Central interior region                             │
│    - 4×8 local texture/color patch grid                  │
└──────────────────────────┬───────────────────────────────┘
                           │ NoteRegionMap
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Module 9 (Optional): Reference-Assisted Comparison       │
│ (src/reference_comparison.py)                            │
│  • ORB feature detection + Lowe's ratio matching         │
│  • RANSAC homography estimation & geometry validation    │
│  • Post-alignment SSIM verification                      │
│  • Automatic fallback if registration is unreliable      │
└──────────────────────────┬───────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┬──────────────────┐
        ▼                  ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Module 5:     │  │ Module 6:     │  │ Module 7:     │  │ Module 8:     │
│ Texture       │  │ Edge & Corner │  │ Morphology &  │  │ Color & Stain │
│ Analysis      │  │ Analysis      │  │ Fold Analysis │  │ Analysis      │
│ (GLCM + LBP)  │  │ (Canny/Sobel) │  │ (Top/BlackHat)│  │ (HSV/CIE Lab) │
└───────┬───────┘  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
        └──────────────────┼──────────────────┴──────────────────┘
                           │ Feature Descriptors & Raw Masks
                           ▼
┌──────────────────────────────────────────────────────────┐
│ Module 10: Damage-Indicator Detection                    │
│ (src/damage_detection.py)                                │
│  • A. Edge wear          • D. Surface texture anomaly    │
│  • B. Corner damage      • E. Stain-like regions         │
│  • C. Fold-like lines    • F. Fading / discoloration     │
└──────────────┬───────────────────────────┬───────────────┘
               │                           │
               ▼                           ▼
┌──────────────────────────────┐ ┌─────────────────────────────────┐
│ Module 11: AMFM Fusion       │ │ Module 18: Damage Localization  │
│ (src/feature_fusion.py)      │ │ (src/damage_localization.py)    │
│  • 0–100 Condition Scores    │ │  • Noise cleanup & CC labeling  │
│  • Capture-quality decoupling│ │  • Color-coded rectified overlay│
│  • Missing-feature tracking  │ │  • Inverse homography projection│
└──────────────┬───────────────┘ └─────────────────┬───────────────┘
               │                                   │
               ▼                                   │
┌──────────────────────────────┐                   │
│ Module 12: CHI Scoring       │                   │
│ (src/health_scoring.py)      │                   │
│  • Weighted fusion / partial │                   │
│    re-normalization          │                   │
│  • Unrounded categorization  │                   │
└──────────────┬───────────────┘                   │
               └─────────────────┬─────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────┐
│ Modules 19 & 20: Reporting, Visualization & Evaluation   │
│ (src/report_generation.py, src/visualization.py, app.py) │
│  • Interactive Streamlit dashboard & Matplotlib charts   │
│  • Downloadable PNG, JSON, CSV, and ReportLab PDF exports│
└──────────────────────────────────────────────────────────┘
```

---

## 3. Module Responsibilities

| File | Responsibility |
| :--- | :--- |
| `config/settings.py` | Central configuration: CHI default weights, category bounds, capture-quality thresholds, region fractions, color legend RGB values, and `AnalysisSettings` dataclass. |
| `src/image_io.py` | Safe loading from paths, bytes, streams, or NumPy arrays; EXIF transposition; RGB/BGR/grayscale/RGBA conversion; file size and dimension validation. |
| `src/image_quality.py` | Computes mean brightness, contrast std, dynamic range, Laplacian variance blur metric, exposure clipping, and spatial illumination non-uniformity; sets `is_provisional`. |
| `src/preprocessing.py` | Aspect-preserving resize, edge-preserving bilateral/Gaussian/median denoising, Lab L-channel illumination normalization, CLAHE enhancement, and HSV/Lab conversions. |
| `src/segmentation.py` | Multi-cue foreground extraction, external contour filtering, quadrilateral envelope fitting, 4-point perspective warp (`homography` and `inv_homography`), and manual crop fallback. |
| `src/region_analysis.py` | Scale-invariant anatomical partitioning into 4 boundary edges, 4 corner regions, central interior region, and a `4×8` local patch grid. |
| `src/texture_analysis.py` | Patch-based and global GLCM (`contrast`, `homogeneity`, `energy`, `correlation`) and uniform LBP (`entropy`, histogram L1 distance, micro-roughness residual). |
| `src/edge_analysis.py` | Separates outer boundary integrity (4 edges + 4 corners) from internal Canny/Sobel edges; measures continuity, missing area fraction, boundary roughness, and corner vertex retention. |
| `src/morphology_analysis.py` | Scale-aware erosion, dilation, opening, closing, morphological gradient, white top-hat, black-hat, directional line opening, probabilistic Hough line extraction, and interior hole detection. |
| `src/color_analysis.py` | Illumination-normalized HSV and CIE Lab analysis for localized stain blotches and regional/global contrast/saturation fading indicators. |
| `src/reference_comparison.py` | ORB keypoint detection, Lowe's ratio matching, RANSAC homography registration, determinant/convexity/area geometry validation, SSIM verification, and structural/Lab difference maps. |
| `src/damage_detection.py` | Six dedicated detector functions returning structured `DamageIndicatorResult` objects with qualitative confidence labels (`High`, `Moderate`, `Low`, `Not reliably assessed`). |
| `src/feature_fusion.py` | **AMFM**: Converts defect indicators into 0–100 component condition scores, applies illumination/resolution noise allowances, and prevents poor capture quality from becoming a physical damage penalty. |
| `src/health_scoring.py` | Computes **CHI**, handles missing components via weight re-normalization or score withholding, and assigns health categories using exact unrounded boundaries. |
| `src/damage_localization.py` | Morphological mask cleanup, connected-component bounding box and centroid extraction, 4-color legend overlay in rectified space, and inverse-homography overlay on the original image. |
| `src/report_generation.py` | Builds the structured report dictionary and exports lossless PNGs, JSON, CSV, and multi-section ReportLab PDF reports. |
| `src/visualization.py` | Generates Matplotlib figures for the CHI gauge spectrum and AMFM component bar chart. |
| `src/evaluation.py` | Computes pixel-level Precision, Recall, F1-score, and IoU against ground-truth masks; generates deterministic synthetic test specimens. |
| `src/pipeline.py` | Orchestrates the end-to-end execution of all modules and tracks total processing time. |
| `app.py` | Streamlit user interface with session-state management, sidebar controls, visual comparison panels, diagnostic tabs, and one-click downloads. |
