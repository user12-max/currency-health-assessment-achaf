"""Central configuration and scoring parameters for ACHAF.

All heuristic thresholds, normalization bounds, and Currency Health Index (CHI)
weights are defined here so they remain inspectable, adjustable, and consistent
across the entire pipeline.

Note: These weights and category boundaries are proposed experimental parameters
for this academic Digital Image Processing project. They do NOT represent
official RBI, banking, or ATM circulation-fitness standards.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Project Metadata
# ---------------------------------------------------------------------------
PROJECT_TITLE: str = (
    "Adaptive Multi-Feature Framework for Currency Note Health Assessment and "
    "Damage Localization using Texture, Edge, and Morphological Analysis"
)
FRAMEWORK_NAME: str = "ACHAF — Adaptive Currency Health Assessment Framework"
CORE_MODULE_NAME: str = "AMFM — Adaptive Multi-Feature Fusion Module"
SCORING_METRIC_NAME: str = "Currency Health Index (CHI)"
PROTOTYPE_DISCLAIMER: str = (
    "Estimated physical-condition category. Prototype academic assessment; "
    "official banking, RBI, or ATM circulation fitness has not been established."
)

SUPPORTED_DENOMINATIONS: List[str] = [
    "Unknown",
    "₹10",
    "₹20",
    "₹50",
    "₹100",
    "₹200",
    "₹500",
]

SUPPORTED_SIDES: List[str] = ["Unknown", "Front", "Back"]

ANALYSIS_MODES: List[str] = [
    "Feature-based analysis",
    "Reference-assisted analysis",
]

ALLOWED_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp")
MAX_FILE_SIZE_MB: float = 15.0
MAX_FILE_SIZE_BYTES: int = int(MAX_FILE_SIZE_MB * 1024 * 1024)


# ---------------------------------------------------------------------------
# Image Validation & Quality Thresholds
# ---------------------------------------------------------------------------
MIN_IMAGE_WIDTH: int = 120
MIN_IMAGE_HEIGHT: int = 60
RECOMMENDED_MIN_WIDTH: int = 400
RECOMMENDED_MIN_HEIGHT: int = 200
MAX_PROCESSING_DIMENSION: int = 1200  # Max width after aspect-preserving resize
STANDARD_RECTIFIED_WIDTH: int = 800
STANDARD_RECTIFIED_HEIGHT: int = 380  # ~2.1:1 typical currency aspect ratio

# Capture quality thresholds
BLUR_LAPLACIAN_WARN_THRESHOLD: float = 45.0
BLUR_LAPLACIAN_SEVERE_THRESHOLD: float = 18.0
BRIGHTNESS_LOW_THRESHOLD: float = 45.0
BRIGHTNESS_HIGH_THRESHOLD: float = 225.0
CONTRAST_LOW_STD_THRESHOLD: float = 22.0
OVEREXPOSED_FRACTION_THRESHOLD: float = 0.25
UNDEREXPOSED_FRACTION_THRESHOLD: float = 0.30


# ---------------------------------------------------------------------------
# Currency Health Index (CHI) Weights & Category Thresholds
# ---------------------------------------------------------------------------
DEFAULT_CHI_WEIGHTS: Dict[str, float] = {
    "edge": 0.30,
    "fold": 0.25,
    "texture": 0.20,
    "color": 0.15,
    "corner": 0.10,
}

# Minimum total weight of available components required to report an overall CHI
MIN_AVAILABLE_WEIGHT_FOR_CHI: float = 0.50

# Category lower bounds (evaluated on unrounded CHI score in descending order):
#   CHI >= 90.0             -> Excellent
#   75.0 <= CHI < 90.0      -> Good
#   60.0 <= CHI < 75.0      -> Fair
#   40.0 <= CHI < 60.0      -> Poor
#   CHI < 40.0              -> Critical
CHI_CATEGORY_BOUNDS: List[Tuple[float, str]] = [
    (90.0, "Excellent"),
    (75.0, "Good"),
    (60.0, "Fair"),
    (40.0, "Poor"),
    (0.0, "Critical"),
]


# ---------------------------------------------------------------------------
# Region Analysis Normalized Fractions
# ---------------------------------------------------------------------------
EDGE_BAND_FRACTION: float = 0.08      # 8% of note height/width for boundary bands
CORNER_BOX_FRACTION: float = 0.16     # 16% of width/height for each corner region
CENTRAL_MARGIN_FRACTION: float = 0.12 # Central region excludes 12% outer border
TEXTURE_GRID_ROWS: int = 4
TEXTURE_GRID_COLS: int = 8


# ---------------------------------------------------------------------------
# Visualization & Damage Localization Color Legend (RGB tuples)
# ---------------------------------------------------------------------------
DAMAGE_COLOR_LEGEND_RGB: Dict[str, Tuple[int, int, int]] = {
    "edge_corner": (220, 38, 38),      # Red: edge or corner irregularity
    "fold_line": (37, 99, 235),        # Blue: fold-like line indicators
    "stain_color": (234, 179, 8),      # Yellow/Amber: stain-like or color anomaly
    "texture": (147, 51, 234),         # Purple: texture irregularity indicators
}

DAMAGE_CATEGORY_LABELS: Dict[str, str] = {
    "edge_corner": "Edge / Corner Irregularity (Red)",
    "fold_line": "Fold-like Line Indicator (Blue)",
    "stain_color": "Stain / Color Anomaly (Yellow)",
    "texture": "Surface Texture Irregularity (Purple)",
}


# ---------------------------------------------------------------------------
# Configurable Pipeline Settings Dataclass
# ---------------------------------------------------------------------------
@dataclass
class AnalysisSettings:
    """User-configurable and adaptive parameters for the ACHAF pipeline."""

    # Preprocessing parameters
    target_max_width: int = MAX_PROCESSING_DIMENSION
    denoise_method: str = "bilateral"  # "gaussian", "median", "bilateral", or "none"
    denoise_strength: int = 5          # Kernel size (odd) or filter diameter
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    apply_illumination_norm: bool = True

    # Segmentation parameters
    enable_auto_segmentation: bool = True
    canny_low_threshold: int = 40
    canny_high_threshold: int = 140
    min_note_area_ratio: float = 0.15  # Minimum contour area relative to image area
    manual_crop_margins: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # (top_pct, bottom_pct, left_pct, right_pct) in [0.0, 0.45]

    # Feature & sensitivity parameters
    fold_sensitivity: float = 1.0      # Multiplier in [0.5, 2.0]
    stain_sensitivity: float = 1.0     # Multiplier in [0.5, 2.0]
    texture_sensitivity: float = 1.0   # Multiplier in [0.5, 2.0]
    edge_sensitivity: float = 1.0      # Multiplier in [0.5, 2.0]
    morphology_scale_factor: float = 1.0

    # Reference alignment parameters
    orb_max_features: int = 1500
    min_inlier_matches: int = 18
    min_inlier_ratio: float = 0.22
    ransac_reproj_threshold: float = 5.0

    # CHI weights
    chi_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CHI_WEIGHTS)
    )
