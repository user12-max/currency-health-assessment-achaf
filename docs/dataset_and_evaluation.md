# Dataset Collection, Annotation, and Quantitative Evaluation Guide

## 1. Practical Dataset Collection Protocol

Because public currency-condition datasets with pixel-level damage masks are scarce, researchers evaluating **ACHAF** on real physical currency notes should follow a standardized acquisition protocol:

1. **Sample Diversity**:
   - Collect genuine currency notes across target denominations (`₹10`, `₹20`, `₹50`, `₹100`, `₹200`, `₹500`) and both sides (`Front`, `Back`).
   - Include notes spanning clean/uncirculated condition, mild circulation wear, prominent central/diagonal creases, localized stains, and corner/edge tears.
2. **Imaging Setup**:
   - Place each note flat on a matte, high-contrast, uniform background (e.g., dark charcoal or matte black board) with at least a $5\text{–}10\%$ visible margin around all four outer edges.
   - Use diffuse, even overhead lighting (avoiding harsh directional flash glare or deep cast shadows across the note).
   - Capture at a minimum resolution of $800 \times 400\text{ px}$ (recommended $1200 \times 600\text{ px}$ or higher) with the camera parallel to the surface (or up to $\pm 20^\circ$ tilt to test perspective rectification).
3. **Reference Image Pairing**:
   - For reference-assisted evaluation, photograph one clean, uncreased note of each denomination and side under the same lighting setup.

---

## 2. Ground-Truth Defect Mask Annotation Procedure

To quantitatively evaluate spatial damage localization:
1. Run the note through ACHAF and download the **Processed Note (PNG)** (`achaf_processed_note.png`, canonical $800 \times H_r$ rectified coordinates).
2. Open the rectified note image in an annotation tool (e.g., CVAT, LabelMe, GIMP, or ImageJ).
3. Create a binary mask image of identical pixel dimensions where:
   - `255` (white) marks ground-truth damaged pixels (missing corners, edge tears, crease lines, stains, or scuffed patches).
   - `0` (black) marks intact note regions.
4. Export the mask as a lossless `.png` file.

---

## 3. Quantitative Evaluation Metrics (`src/evaluation.py`)

When a ground-truth binary mask $G \in \{0, 1\}^{H \times W}$ is supplied alongside the predicted localization mask $P \in \{0, 1\}^{H \times W}$, ACHAF computes:

- **True Positives ($\text{TP}$)**: $\sum_{x,y} \mathbb{I}(P(x,y)=1 \land G(x,y)=1)$
- **False Positives ($\text{FP}$)**: $\sum_{x,y} \mathbb{I}(P(x,y)=1 \land G(x,y)=0)$
- **False Negatives ($\text{FN}$)**: $\sum_{x,y} \mathbb{I}(P(x,y)=0 \land G(x,y)=1)$
- **Precision**:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **Recall**:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **F1-Score (Dice Coefficient)**:
  $$\text{F1} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Intersection over Union (IoU / Jaccard Index)**:
  $$\text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
- **Processing Time**: End-to-end CPU execution latency in milliseconds ($\text{ms}$).

### How to Run Evaluation in the Dashboard
1. Analyze a note in the Streamlit app (`app.py`).
2. Open **Section 4 → Tab 6 (Reference & Evaluation)**.
3. Upload your binary ground-truth mask under **Quantitative Ground-Truth Mask Evaluation** to view real-time Precision, Recall, F1-Score, and IoU metrics.
