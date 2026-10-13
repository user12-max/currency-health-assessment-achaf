"""Module 4: Region Analysis.

Partitions a rectified currency note into scale-invariant, normalized anatomical
regions:
- Four boundary edges (top, bottom, left, right).
- Four corner regions (top-left, top-right, bottom-left, bottom-right).
- Central interior region.
- Uniform grid of local texture/color patches.

All coordinates scale proportionally with the image dimensions so that no fixed
pixel thresholds are assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from config.settings import (
    CENTRAL_MARGIN_FRACTION,
    CORNER_BOX_FRACTION,
    EDGE_BAND_FRACTION,
    TEXTURE_GRID_COLS,
    TEXTURE_GRID_ROWS,
)


@dataclass
class BoundingBoxRegion:
    """Normalized and pixel-level bounding box for a specific note sub-region."""

    name: str
    category: str  # "edge", "corner", "central", or "patch"
    x1: int
    y1: int
    x2: int
    y2: int
    norm_bbox: Tuple[float, float, float, float]  # (x1/W, y1/H, x2/W, y2/H)

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    def extract_slice(self, array: np.ndarray) -> np.ndarray:
        """Slice the region from a 2D or 3D image array."""
        return array[self.y1 : self.y2, self.x1 : self.x2]

    def create_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Create a full-frame binary uint8 mask (255 inside region, 0 outside)."""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[self.y1 : self.y2, self.x1 : self.x2] = 255
        return mask


# Alias with clean name
RegionBox = BoundingBoxRegion


@dataclass
class NoteRegionMap:
    """Complete anatomical partition of a rectified currency note."""

    image_width: int
    image_height: int
    edge_regions: Dict[str, RegionBox]    # "top", "bottom", "left", "right"
    corner_regions: Dict[str, RegionBox]  # "top_left", "top_right", "bottom_left", "bottom_right"
    central_region: RegionBox
    grid_patches: List[RegionBox]
    grid_rows: int
    grid_cols: int

    def locate_point_region(self, x: float, y: float) -> str:
        """Return a human-readable anatomical label for pixel coordinate (x, y)."""
        for name, box in self.corner_regions.items():
            if box.x1 <= x < box.x2 and box.y1 <= y < box.y2:
                return f"Corner ({name.replace('_', '-')})"
        for name, box in self.edge_regions.items():
            if box.x1 <= x < box.x2 and box.y1 <= y < box.y2:
                return f"Edge ({name})"
        return "Central / Interior surface"


def partition_note_regions(
    image_shape: Tuple[int, ...],
    edge_fraction: float = EDGE_BAND_FRACTION,
    corner_fraction: float = CORNER_BOX_FRACTION,
    central_margin: float = CENTRAL_MARGIN_FRACTION,
    grid_rows: int = TEXTURE_GRID_ROWS,
    grid_cols: int = TEXTURE_GRID_COLS,
) -> NoteRegionMap:
    """Partition an image of shape (H, W[, C]) into normalized note regions.

    Parameters
    ----------
    image_shape : tuple
        Shape of the rectified note image `(height, width, ...)`.
    edge_fraction : float
        Fraction of note dimensions dedicated to each boundary edge strip.
    corner_fraction : float
        Fraction of note dimensions dedicated to each corner box.
    central_margin : float
        Outer margin fraction excluded from the central region.
    grid_rows, grid_cols : int
        Dimensions of the local patch grid.

    Returns
    -------
    NoteRegionMap
        Structured region boxes and lookup utilities.
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    if h < 16 or w < 16:
        raise ValueError(f"Image dimensions ({w}x{h}) are too small for region partitioning.")

    edge_h = max(4, int(round(h * np.clip(edge_fraction, 0.03, 0.20))))
    edge_w = max(4, int(round(w * np.clip(edge_fraction, 0.03, 0.20))))

    corner_h = max(6, int(round(h * np.clip(corner_fraction, 0.08, 0.30))))
    corner_w = max(6, int(round(w * np.clip(corner_fraction, 0.08, 0.30))))

    margin_h = max(4, int(round(h * np.clip(central_margin, 0.05, 0.30))))
    margin_w = max(4, int(round(w * np.clip(central_margin, 0.05, 0.30))))

    def _make_box(name: str, category: str, x1: int, y1: int, x2: int, y2: int) -> RegionBox:
        x1_c = max(0, min(w - 1, x1))
        y1_c = max(0, min(h - 1, y1))
        x2_c = max(x1_c + 1, min(w, x2))
        y2_c = max(y1_c + 1, min(h, y2))
        return RegionBox(
            name=name,
            category=category,
            x1=x1_c,
            y1=y1_c,
            x2=x2_c,
            y2=y2_c,
            norm_bbox=(x1_c / w, y1_c / h, x2_c / w, y2_c / h),
        )

    # Four corner regions
    corner_regions: Dict[str, RegionBox] = {
        "top_left": _make_box("top_left", "corner", 0, 0, corner_w, corner_h),
        "top_right": _make_box("top_right", "corner", w - corner_w, 0, w, corner_h),
        "bottom_left": _make_box("bottom_left", "corner", 0, h - corner_h, corner_w, h),
        "bottom_right": _make_box(
            "bottom_right", "corner", w - corner_w, h - corner_h, w, h
        ),
    }

    # Four boundary edge strips (excluding corner boxes so edge wear and corner damage
    # are not double-counted)
    edge_regions: Dict[str, RegionBox] = {
        "top": _make_box("top", "edge", corner_w, 0, w - corner_w, edge_h),
        "bottom": _make_box(
            "bottom", "edge", corner_w, h - edge_h, w - corner_w, h
        ),
        "left": _make_box("left", "edge", 0, corner_h, edge_w, h - corner_h),
        "right": _make_box(
            "right", "edge", w - edge_w, corner_h, w, h - corner_h
        ),
    }

    # Central interior region
    central_region = _make_box(
        "central", "central", margin_w, margin_h, w - margin_w, h - margin_h
    )

    # Local texture/color patch grid covering the note
    rows = max(2, int(grid_rows))
    cols = max(2, int(grid_cols))
    grid_patches: List[RegionBox] = []
    row_edges = np.linspace(0, h, rows + 1, dtype=int)
    col_edges = np.linspace(0, w, cols + 1, dtype=int)

    for r in range(rows):
        for c in range(cols):
            patch_name = f"patch_r{r}_c{c}"
            grid_patches.append(
                _make_box(
                    patch_name,
                    "patch",
                    int(col_edges[c]),
                    int(row_edges[r]),
                    int(col_edges[c + 1]),
                    int(row_edges[r + 1]),
                )
            )

    return NoteRegionMap(
        image_width=w,
        image_height=h,
        edge_regions=edge_regions,
        corner_regions=corner_regions,
        central_region=central_region,
        grid_patches=grid_patches,
        grid_rows=rows,
        grid_cols=cols,
    )
