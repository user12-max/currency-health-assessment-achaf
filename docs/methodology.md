# ACHAF Digital Image Processing Methodology

## 1. Scope & Academic Positioning

**ACHAF (Adaptive Currency Health Assessment Framework)** is an academic Digital Image Processing (DIP) prototype designed to assess visual indicators of physical deterioration on genuine currency notes and localize affected regions spatially.

- **Not Counterfeit Detection**: ACHAF does not authenticate security features, watermarks, micro-lettering, or serial numbers.
- **Not Official Fitness Certification**: The **Currency Health Index (CHI)** weights and category thresholds are proposed experimental parameters for academic DIP study—not official Reserve Bank of India (RBI), central bank, or ATM sorting standards.

---

## 2. Capture-Quality Assessment & Preprocessing

### 2.1 Capture-Quality Decoupling (`src/image_quality.py`)
Photographic capture flaws must not be conflated with physical note degradation. Before feature extraction, ACHAF computes:
- **Mean Brightness & Exposure Clipping**: Mean grayscale intensity $\mu_I \in [0, 255]$, overexposed pixel fraction ($I \ge 250$), and underexposed pixel fraction ($I \le 10$).
- **Global Contrast**: Standard deviation $\sigma_I$ and inter-percentile dynamic range $P_{95} - P_5$.
- **Optical Sharpness / Blur Indicator**: Variance of the discrete Laplacian $\text{Var}(\nabla^2 I)$ evaluated at a standardized maximum dimension ($\le 800\text{ px}$).
  - $\text{Var}(\nabla^2 I) < 45$: Moderate blur warning.
  - $\text{Var}(\nabla^2 I) < 18$: Severe blur; fine fold and surface texture channels are marked **Not reliably assessed** rather than penalized.
- **Illumination Non-Uniformity**: Coefficient of variation of block-wise mean intensities across a $4 \times 4$ spatial grid.

### 2.2 Preprocessing (`src/preprocessing.py`)
1. **Aspect-Preserving Scaling**: Downscales images exceeding `MAX_PROCESSING_DIMENSION` ($1200\text{ px}$) using area interpolation (`cv2.INTER_AREA`) while recording exact scale factors $(s_x, s_y)$.
2. **Detail-Preserving Denoising**: Applies bilateral filtering (`cv2.bilateralFilter`, default $d=5, \sigma_c=25, \sigma_s=25$) so sensor noise is attenuated without erasing thin fold lines or small edge nicks.
3. **Illumination Normalization**: Converts RGB to CIE $L^*a^*b^*$, estimates the low-frequency illumination envelope $L_{\text{bg}}$ via a large-kernel Gaussian filter ($k \approx 0.25 \max(W, H)$), normalizes $L^*$, and blends $65\%$ normalized + $35\%$ original luminance to flatten broad shadow gradients while preserving localized stain contrast.
4. **CLAHE Enhancement**: Applies Contrast Limited Adaptive Histogram Equalization (`clipLimit=2.0, tileGridSize=(8,8)`) for keypoint detection and diagnostic inspection.

---

## 3. Currency Note Segmentation & Perspective Rectification (`src/segmentation.py`)

1. **Multi-Cue Foreground Masking**:
   - Models the outer frame border ($2.5\%$ margin) in CIE $L^*a^*b^*$ space and thresholds Euclidean color distance from the background median.
   - Combines border-color segmentation with Otsu thresholding and closed Canny edge contours (`cv2.RETR_EXTERNAL`).
2. **Quadrilateral Envelope Fitting**:
   - Extracts the largest external contour and verifies that its area ratio exceeds `min_note_area_ratio` ($15\%$).
   - Fits a 4-vertex polygon via convex-hull Douglas-Peucker approximation (`cv2.approxPolyDP`) or minimum-area bounding rectangle (`cv2.minAreaRect`). Using the bounding quadrilateral envelope ensures that missing corners and edge tears remain inside the rectified coordinate frame where they can be measured as foreground deficits.
3. **Perspective Homography**:
   - Orders corners consistently as $[\text{TL}, \text{TR}, \text{BR}, \text{BL}]$ and computes both the forward perspective matrix $H \in \mathbb{R}^{3\times 3}$ and inverse matrix $H^{-1} \in \mathbb{R}^{3\times 3}$ onto a canonical frame ($800 \times H_r$).
4. **Safe Fallback**:
   - When no valid contour is isolated, `segmentation_succeeded` is set to `False` and the pipeline falls back to configurable manual crop margins without fabricating perspective rectification.

---

## 4. Multi-Domain Feature Extraction

### 4.1 Normalized Anatomical Region Partitioning (`src/region_analysis.py`)
All region coordinates scale proportionally with rectified dimensions $(W_r, H_r)$:
- **Four Corner Boxes**: $16\% W_r \times 16\% H_r$ at `top_left`, `top_right`, `bottom_left`, and `bottom_right`.
- **Four Boundary Edge Strips**: $8\%$ outer bands along `top`, `bottom`, `left`, and `right` (excluding corner boxes to prevent double-counting).
- **Central Interior Region**: Interior rectangle excluding the outer $12\%$ perimeter.
- **Local Patch Grid**: $4 \times 8$ uniform grid ($32$ patches) for localized texture and chromatic comparisons.

### 4.2 Texture Analysis (`src/texture_analysis.py`)
- **Gray Level Co-occurrence Matrix (GLCM)**: Quantizes intensities to $32$ levels and computes `contrast`, `homogeneity`, `energy`, and `correlation` across distances $d \in \{1, 2\}$ and orientations $\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$. Uniform patches ($\sigma \approx 0$) are handled explicitly to avoid division-by-zero `NaN` in correlation.
- **Local Binary Patterns (LBP)**: Computes rotation-aware uniform LBP ($P=8, R=1$) and Shannon entropy of the normalized 10-bin LBP histogram.
- **Print-Pattern vs. Physical Wear Separation**: Because currency notes naturally contain heterogeneous printed patterns, single-image mode evaluates isotropic micro-roughness residuals (difference between median-filtered and bilateral-filtered grayscale) alongside GLCM homogeneity. When a reliably aligned reference note is available, homologous grid patches are compared directly via LBP histogram $L_1$ distance and GLCM homogeneity difference.

### 4.3 Edge and Corner Analysis (`src/edge_analysis.py`)
- **External Boundary Integrity**: Evaluates each of the 4 edge strips for:
  - Missing area fraction ($1 - \text{foreground coverage}$).
  - Boundary continuity ratio (fraction of scanlines reaching the expected outer edge within a $2\text{ px}$ tolerance).
  - Boundary displacement roughness (normalized standard deviation of edge profile coordinates).
- **Corner Geometry Integrity**: Evaluates each of the 4 corner boxes for foreground area retention and normalized Euclidean distance from the ideal corner vertex to the nearest note pixel.
- **Internal Gradient Separation**: Computes Canny edges and Sobel gradient magnitude $(\sqrt{G_x^2 + G_y^2})$ inside the central region strictly for diagnostic inspection, never confusing internal printed lines with torn outer boundaries.

### 4.4 Scale-Aware Morphological & Fold-Line Analysis (`src/morphology_analysis.py`)
- Computes scale-normalized erosion, dilation, opening, closing, morphological gradient, white top-hat, and black-hat maps ($k \approx 1.5\% \min(W_r, H_r)$).
- **Coupled Trough-Ridge Directional Crease Enhancement**:
  - Disconnected printed text characters have gaps between letters; applying directional morphological opening (`cv2.MORPH_OPEN`) with 8 oriented line structuring elements ($\theta \in \{0^\circ, 22.5^\circ, \dots, 157.5^\circ\}$) directly to the thin dark trough map breaks character strings while preserving continuous creases.
  - Couples continuous dark troughs with adjacent bright highlight/wear ridges above the local paper baseline to suppress flat absorptive security threads.
  - Extracts elongated crease segments via Probabilistic Hough Transform (`cv2.HoughLinesP`) with collinear segment deduplication.

### 4.5 Color, Stain, and Fading Analysis (`src/color_analysis.py`)
- Operates on illumination-normalized HSV and CIE $L^*a^*b^*$ channels inside the note interior.
- **Stain Detection**: Computes medium-scale Gaussian-smoothed chromatic distance $\Delta_{ab} = \sqrt{(a^* - \tilde{a}^*)^2 + (b^* - \tilde{b}^*)^2}$ from the note's robust median palette and localized luminance drops $\Delta L^*$, filtering out narrow full-height security threads.
- **Fading / Discoloration**: Evaluates luminance contrast loss and saturation reduction (in reference-assisted mode via direct CIE $\Delta E^*_{ab}$ and contrast comparison; in single-image mode as a low-confidence approximate indicator).

### 4.6 Reference-Assisted Alignment (`src/reference_comparison.py`)
- Extracts up to $1500$ ORB keypoints on CLAHE-enhanced grayscale images, filters descriptor matches via Lowe's ratio test ($0.78$), and estimates a planar homography $H$ via RANSAC.
- **Validation Checks**: Requires $\ge 18$ inliers, $\ge 22\%$ inlier ratio, plausible Jacobian determinant ($0.15 \le \det(H_{2\times 2}) \le 6.5$), convex warped corners, compatible area ratio, and post-warp Structural Similarity $\text{SSIM} \ge 0.22$. Failed registrations are rejected with a clear explanation and automatic fallback to feature-based mode.

---

## 5. Adaptive Multi-Feature Fusion (AMFM) & Currency Health Index (CHI)

### 5.1 Component Condition Conversion (`src/feature_fusion.py`)
Each feature channel $i \in \{\text{edge}, \text{fold}, \text{texture}, \text{color}, \text{corner}\}$ converts its normalized defect severity $D_i \in [0, 100]$ into a condition score $S_i \in [0, 100]$:
$$S_i = \text{clip}\left(100 - \max(0, D_i - \tau_{\text{noise}, i}),\; 0,\; 100\right)$$
where $\tau_{\text{noise}, i}$ is an adaptive allowance for illumination non-uniformity or low resolution. If severe blur or extreme exposure clipping prevents reliable evaluation of channel $i$, $S_i$ is marked **Unavailable** (`None`) rather than assigned an arbitrary score.

### 5.2 CHI Formula & Missing-Component Handling (`src/health_scoring.py`)
When all 5 components are available, CHI is calculated using the default configurable weights:
$$\text{CHI} = 0.30\,S_{\text{edge}} + 0.25\,S_{\text{fold}} + 0.20\,S_{\text{texture}} + 0.15\,S_{\text{color}} + 0.10\,S_{\text{corner}}$$

When a subset of components $\mathcal{A}$ is available with total base weight $W_{\mathcal{A}} = \sum_{i \in \mathcal{A}} w_i$:
- If $W_{\mathcal{A}} \ge 0.50$: CHI is computed via re-normalized weights $\tilde{w}_i = w_i / W_{\mathcal{A}}$ and flagged as **Normalized Partial-Feature Weighted Fusion (Provisional)**.
- If $W_{\mathcal{A}} < 0.50$: Overall CHI is **Withheld** (`Unable to Assess`).

### 5.3 Category Assignment Boundaries
Evaluated on the exact unrounded floating-point CHI value:
- $90.0 \le \text{CHI} \le 100.0$: **Excellent**
- $75.0 \le \text{CHI} < 90.0$: **Good**
- $60.0 \le \text{CHI} < 75.0$: **Fair**
- $40.0 \le \text{CHI} < 60.0$: **Poor**
- $0.0 \le \text{CHI} < 40.0$: **Critical**
