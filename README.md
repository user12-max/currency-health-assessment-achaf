# ACHAF — Adaptive Currency Health Assessment Framework

**Project Title:** Adaptive Multi-Feature Framework for Currency Note Health Assessment and Damage Localization using Texture, Edge, and Morphological Analysis  
**Core Fusion Module:** AMFM — Adaptive Multi-Feature Fusion Module  
**Assessment Metric:** Currency Health Index (CHI)

---

## 1. Project Overview

**ACHAF** is a complete, CPU-runnable Digital Image Processing (DIP) application built in Python for evaluating physical deterioration indicators on photographs of currency notes, localizing damaged regions spatially, and generating explainable multi-format health reports.

The application analyzes six visual deterioration indicators using traditional, explainable computer-vision algorithms:
1. **Edge Wear & Boundary Irregularities** (4-edge contour continuity, missing border area fraction, and boundary displacement roughness).
2. **Missing or Damaged Corners** (4-corner area retention and vertex tip geometry).
3. **Fold-like & Crease-like Lines** (scale-aware morphological top-hat/black-hat coupled ridge-trough filtering, 8-orientation directional line opening, and probabilistic Hough line detection).
4. **Surface Texture Irregularities** (patch-based Gray Level Co-occurrence Matrix [GLCM] and uniform Local Binary Patterns [LBP] with micro-roughness separation).
5. **Stain-like Regions & Localized Color Anomalies** (illumination-normalized HSV and CIE $L^*a^*b^*$ chromatic/luminance outlier detection).
6. **Possible Fading or Discoloration** (regional contrast and saturation loss, optionally enhanced via ORB+RANSAC reference-note comparison).

> **Academic Scope & Disclaimer:** This is an academic Digital Image Processing prototype focused on physical-condition assessment—**NOT** counterfeit detection, denomination recognition, serial-number OCR, or official banking/RBI/ATM fitness certification.

---

## 2. Main Features

- **Robust Image Acquisition & Validation (`src/image_io.py`)**: Supports `PNG`, `JPG`, `JPEG`, and `WebP` up to `15 MB`; handles `RGB`, `BGR`, `Grayscale`, and transparent `RGBA` images; applies automatic EXIF orientation correction; rejects empty, corrupted, or undersized inputs safely.
- **Capture-Quality Assessment (`src/image_quality.py`)**: Evaluates brightness, contrast, exposure clipping, spatial illumination uniformity, and Laplacian-variance sharpness separately from physical note health, marking poor captures as **Provisional** rather than penalizing the note's condition score.
- **Note Segmentation & Perspective Rectification (`src/segmentation.py`)**: Multi-cue foreground extraction (border CIE $L^*a^*b^*$ modeling, Otsu thresholding, Canny edges, contour approximation), 4-corner quadrilateral fitting, forward/inverse homography warping, and a configurable **Manual Crop Fallback**.
- **Reference-Assisted Comparison (`src/reference_comparison.py`)**: Optional alignment against a cleaner note of the same denomination and side using ORB features, Lowe's ratio test, and RANSAC homography with strict geometric/SSIM validation and automatic fallback to feature-based mode when alignment is unreliable.
- **Adaptive Multi-Feature Fusion Module (`src/feature_fusion.py`) & Currency Health Index (`src/health_scoring.py`)**: Converts normalized indicators into `0–100` component condition scores, tracks unavailable features honestly (re-normalizing weights or withholding CHI when insufficient evidence exists), and assigns categories using exact unrounded thresholds.
- **Spatial Damage Localization (`src/damage_localization.py`)**: Color-coded overlays in both rectified note coordinates and original perspective coordinates (mapped back via inverse homography $H^{-1}$).
- **Multi-Format Report Exports (`src/report_generation.py`)**: One-click downloads for Annotated Damage-Map `PNG`, Processed Note `PNG`, Structured `JSON` report, Summary `CSV`, and multi-section ReportLab `PDF` report.

---

## 3. Technology Stack & Prerequisites

- **Python**: Python 3.11+ (tested on Python 3.14, 64-bit Windows)
- **UI Dashboard**: `streamlit`
- **Image Processing & Vision**: `opencv-python` (`cv2`), `scikit-image`, `Pillow`, `scipy`
- **Numerical & Tabular**: `numpy`, `pandas`
- **Visualization & PDF Reporting**: `matplotlib`, `reportlab`
- **Automated Testing**: `pytest`

No GPU, external API, cloud service, or deep-learning weights are required.

---

## 4. Installation & Startup Instructions (Windows PowerShell)

Open **Windows PowerShell** in the project root directory (`c:\dip project`) and run:

```powershell
# 1. Create a virtual environment (if not already created)
python -m venv .venv

# 2. Activate the virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1

# 3. Install required dependencies
python -m pip install -r requirements.txt

# 4. Launch the Streamlit application
python -m streamlit run app.py
```

### Handling PowerShell Execution Policy Restrictions
If `.\.venv\Scripts\Activate.ps1` displays a script execution policy error on Windows PowerShell, you do **not** need to disable system-wide security controls. Choose either of these safe options:
- **Option A (Process-scoped bypass for the current terminal window only):**
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```
- **Option B (Run directly via the virtual environment's Python executable without activating):**
  ```powershell
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe -m streamlit run app.py
  ```

### Expected Local URL
Once started, Streamlit serves the application locally at:
- **Local URL:** `http://localhost:8501`

---

## 5. How to Use the Application

1. **Select Input Source (Sidebar → Section 1)**:
   - Choose **Upload Currency Image** to upload a photograph (`.png`, `.jpg`, `.jpeg`, `.webp`) of a currency note, OR select one of the built-in **Synthetic Specimens** (`Clean Note`, `Damaged Note`, or `Angled & Damaged`) to inspect pipeline behavior immediately.
2. **Select Metadata & Analysis Mode (Sidebar → Section 2)**:
   - Choose the **Denomination** (`₹10`, `₹20`, `₹50`, `₹100`, `₹200`, `₹500`, or `Unknown`) and **Visible Side** (`Front`, `Back`, or `Unknown`).
   - Select **Feature-based analysis** (single image) or **Reference-assisted analysis** (optional cleaner reference note of the same denomination, side, and design).
3. **Optional Advanced Settings**:
   - Expand **Advanced Settings** in the sidebar to adjust denoising filter parameters, CLAHE clip limit, Canny thresholds, manual crop margins (used when automatic background segmentation is disabled or falls back), feature sensitivities, and CHI component weights.
4. **Run Analysis**:
   - Click **🔍 Analyze Currency Note**. The app processes the image on demand and displays the complete dashboard.
   - Click **🗑️ Reset / Clear Analysis** at any time to clear current results.

---

## 6. How the Currency Health Index (CHI) Is Calculated

Each of the 5 AMFM feature channels produces a normalized condition score $S_i \in [0, 100]$, where $100$ represents an intact, unblemished condition:

$$\text{CHI} = 0.30 \times S_{\text{edge}} + 0.25 \times S_{\text{fold}} + 0.20 \times S_{\text{texture}} + 0.15 \times S_{\text{color}} + 0.10 \times S_{\text{corner}}$$

- **Missing / Unassessable Components**: If severe capture flaws (e.g., severe optical blur) prevent reliable assessment of a component, that component is marked `Unavailable` (`None`) rather than given a fabricated `100` or `0`. If available components represent $\ge 50\%$ of the total weight, CHI is calculated using re-normalized weights across the available components and labeled accordingly; otherwise, the overall CHI is withheld.
- **Proposed Academic Category Thresholds** (evaluated on the unrounded score):
  - **90 – 100**: `Excellent`
  - **75 – 89** ($75.0 \le \text{CHI} < 90.0$): `Good`
  - **60 – 74** ($60.0 \le \text{CHI} < 75.0$): `Fair`
  - **40 – 59** ($40.0 \le \text{CHI} < 60.0$): `Poor`
  - **Below 40** ($\text{CHI} < 40.0$): `Critical`

---

## 7. Interpreting Outputs & Color Legend

The spatial damage localization overlays use a consistent 4-category color legend:
- **Red (`#DC2626`)**: Edge wear or missing/damaged corner irregularity.
- **Blue (`#2563EB`)**: Fold-like or crease-like line indicators and interior structural gaps.
- **Yellow (`#EAB308`)**: Stain-like blotches, chromatic outliers, or discoloration regions.
- **Purple (`#9333EA`)**: Localized surface texture irregularity (crumple/abrasion roughness).

---

## 8. Running Automated Tests

Run the full `pytest` suite from the project root:

```powershell
python -m pytest -v
```

The test suite (`tests/`) verifies:
- Image loading, EXIF orientation, grayscale/RGBA transparency, and invalid/empty input handling (`test_image_io.py`).
- Aspect-preserving resize, blur/exposure capture quality checks, denoising, illumination normalization, and CLAHE (`test_preprocessing.py`).
- Quadrilateral ordering, perspective rectification on flat and angled notes, and manual crop fallback (`test_segmentation.py`).
- Region partitioning, NaN-safe GLCM & LBP descriptors, 4-edge/4-corner analysis, coupled directional fold detection, Lab stain analysis, and ORB+RANSAC reference alignment (`test_feature_extraction.py`).
- Exact CHI boundary cases (`100, 90, 89, 75, 74, 60, 59, 40, 39, 0`), weighted formula, and missing-component handling (`test_health_scoring.py`).
- Damage localization overlay dimensions, bounding boxes, inverse homography projection, and Precision/Recall/F1/IoU metrics (`test_damage_localization.py`).
- End-to-end JSON, CSV, PNG, and ReportLab PDF export validity (`test_report_generation.py`).

---

## 9. Known Limitations

1. **2D Optical Ambiguity**: A 2D RGB photograph cannot directly measure 3D paper thickness, tensile crispness, or soil mass. Creases that have been pressed completely flat under uniform frontal lighting may exhibit minimal optical shadow/ridge contrast.
2. **Background Contrast for Segmentation**: Automatic quadrilateral segmentation works best when the currency note is photographed against a contrasting, uncluttered background. Cluttered backgrounds may require adjusting the **Manual Crop Margins** in the sidebar.
3. **Single-Image Fading vs. Illumination**: Without an aligned clean reference note captured under comparable lighting, global ink fading cannot be separated with high certainty from camera white-balance shifts or overexposure.
4. **Reference Alignment Constraints**: Reference-assisted comparison requires a reference image of the exact same denomination, side, and design series. Severe perspective distortion, heavy occlusion, or mismatched series will cause RANSAC/SSIM validation to reject registration and fall back to feature-based analysis.

---

## 10. Additional Documentation

- [`docs/architecture.md`](docs/architecture.md) — Module architecture and pipeline data flow.
- [`docs/methodology.md`](docs/methodology.md) — Detailed mathematical and algorithmic descriptions.
- [`docs/dataset_and_evaluation.md`](docs/dataset_and_evaluation.md) — Dataset collection protocol, ground-truth mask annotation, and Precision/Recall/F1/IoU evaluation guide.
- [`sample_data/README.md`](sample_data/README.md) — Explanation of synthetic test specimens generated in `sample_data/`.
