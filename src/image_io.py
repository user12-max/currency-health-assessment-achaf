"""Module 1: Image Acquisition and Validation.

Handles loading currency note images from bytes, file-like objects, file paths,
or NumPy arrays. Performs format verification, EXIF orientation correction,
color-mode normalization (RGB, BGR, grayscale, RGBA/LA transparency), dimension
checks, and file-size validation without crashing on malformed inputs.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config.settings import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
    RECOMMENDED_MIN_HEIGHT,
    RECOMMENDED_MIN_WIDTH,
)


@dataclass
class ImageLoadResult:
    """Structured result of loading and validating an input image."""

    is_valid: bool
    rgb_image: Optional[np.ndarray] = None
    bgr_image: Optional[np.ndarray] = None
    alpha_mask: Optional[np.ndarray] = None
    width: int = 0
    height: int = 0
    channels: int = 0
    original_mode: str = "Unknown"
    format_name: str = "Unknown"
    file_size_bytes: int = 0
    exif_transposed: bool = False
    has_transparency: bool = False
    filename: str = "uploaded_image"
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


def _extract_bytes_and_name(
    source: Union[str, Path, bytes, io.BytesIO, Any],
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Tuple[Optional[bytes], str, Optional[str]]:
    """Extract raw bytes and filename from various input sources safely."""
    filename = "image"
    try:
        if isinstance(source, (str, Path)):
            path_obj = Path(source)
            filename = path_obj.name
            if not path_obj.exists() or not path_obj.is_file():
                return None, filename, f"File does not exist: {filename}"
            size = path_obj.stat().st_size
            if size == 0:
                return None, filename, "The image file is empty (0 bytes)."
            if size > max_file_size_bytes:
                return (
                    None,
                    filename,
                    f"File size ({size / (1024 * 1024):.2f} MB) exceeds the "
                    f"maximum allowed limit of {max_file_size_bytes / (1024 * 1024):.1f} MB.",
                )
            return path_obj.read_bytes(), filename, None

        if isinstance(source, bytes):
            if len(source) == 0:
                return None, filename, "Provided image byte buffer is empty (0 bytes)."
            if len(source) > max_file_size_bytes:
                return (
                    None,
                    filename,
                    f"Image buffer size ({len(source) / (1024 * 1024):.2f} MB) exceeds "
                    f"the maximum limit of {max_file_size_bytes / (1024 * 1024):.1f} MB.",
                )
            return source, filename, None

        # File-like object (e.g., Streamlit UploadedFile or io.BytesIO)
        if hasattr(source, "name") and isinstance(getattr(source, "name"), str):
            filename = os.path.basename(getattr(source, "name"))
        if hasattr(source, "getvalue"):
            raw = source.getvalue()
        elif hasattr(source, "read"):
            if hasattr(source, "seek"):
                source.seek(0)
            raw = source.read()
            if hasattr(source, "seek"):
                source.seek(0)
        else:
            return None, filename, "Unsupported input source type."

        if not isinstance(raw, (bytes, bytearray)) or len(raw) == 0:
            return None, filename, "The uploaded file is empty (0 bytes)."
        if len(raw) > max_file_size_bytes:
            return (
                None,
                filename,
                f"Uploaded file size ({len(raw) / (1024 * 1024):.2f} MB) exceeds "
                f"the maximum allowed limit of {max_file_size_bytes / (1024 * 1024):.1f} MB.",
            )
        return bytes(raw), filename, None
    except Exception as exc:
        return None, filename, f"Failed to read image source: {exc}"


def load_and_validate_image(
    source: Union[str, Path, bytes, io.BytesIO, np.ndarray, Any],
    max_file_size_mb: float = MAX_FILE_SIZE_MB,
    input_color_order: str = "RGB",
) -> ImageLoadResult:
    """Load, orient, normalize, and validate an input image.

    Parameters
    ----------
    source : str | Path | bytes | BytesIO | np.ndarray | UploadedFile
        Input image source.
    max_file_size_mb : float
        Maximum allowed file size in megabytes.
    input_color_order : str
        When `source` is a 3-channel NumPy array, specifies whether its channels
        are ordered as 'RGB' or 'BGR'.

    Returns
    -------
    ImageLoadResult
        Validated image data with both RGB and BGR uint8 arrays, alpha mask if
        applicable, metadata, warnings, and error messages.
    """
    max_bytes = int(max_file_size_mb * 1024 * 1024)
    warnings: List[str] = []

    # Case 1: Direct NumPy array input (used in unit tests or programmatic calls)
    if isinstance(source, np.ndarray):
        return _validate_numpy_image(source, input_color_order=input_color_order)

    # Case 2: File path, bytes, or stream
    raw_bytes, filename, err = _extract_bytes_and_name(source, max_bytes)
    if err is not None or raw_bytes is None:
        return ImageLoadResult(
            is_valid=False,
            filename=filename,
            error_message=err or "Unable to read image bytes.",
        )

    ext = os.path.splitext(filename)[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        return ImageLoadResult(
            is_valid=False,
            filename=filename,
            file_size_bytes=len(raw_bytes),
            error_message=(
                f"Unsupported file extension '{ext}'. "
                f"Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}."
            ),
        )

    try:
        bio = io.BytesIO(raw_bytes)
        with Image.open(bio) as pil_img:
            format_name = (pil_img.format or "Unknown").upper()
            if format_name not in ("PNG", "JPEG", "JPG", "WEBP", "MPO"):
                return ImageLoadResult(
                    is_valid=False,
                    filename=filename,
                    file_size_bytes=len(raw_bytes),
                    format_name=format_name,
                    error_message=(
                        f"Unsupported image format '{format_name}'. "
                        "Please upload a PNG, JPG/JPEG, or WebP image."
                    ),
                )

            # Force full decoding to catch truncated or corrupted files early
            pil_img.load()
            original_mode = pil_img.mode

            # Apply EXIF orientation correction if orientation tag is present
            exif_transposed = False
            try:
                exif = pil_img.getexif()
                orientation = exif.get(0x0112, 1) if exif else 1
                if orientation and orientation != 1:
                    pil_img = ImageOps.exif_transpose(pil_img)
                    exif_transposed = True
                    warnings.append(
                        f"Applied automatic EXIF orientation correction (tag={orientation})."
                    )
            except Exception:
                # If EXIF parsing fails, proceed with raw pixel orientation
                pass

            rgb_arr, alpha_mask, has_transparency, mode_warnings = _convert_pil_to_rgb(
                pil_img
            )
            warnings.extend(mode_warnings)

    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        return ImageLoadResult(
            is_valid=False,
            filename=filename,
            file_size_bytes=len(raw_bytes),
            error_message=f"Invalid, corrupted, or truncated image file: {exc}",
        )
    except Exception as exc:
        return ImageLoadResult(
            is_valid=False,
            filename=filename,
            file_size_bytes=len(raw_bytes),
            error_message=f"Unexpected error while decoding image: {exc}",
        )

    height, width = rgb_arr.shape[:2]
    dim_ok, dim_err, dim_warns = _check_dimensions_and_content(rgb_arr)
    warnings.extend(dim_warns)

    if not dim_ok:
        return ImageLoadResult(
            is_valid=False,
            filename=filename,
            width=width,
            height=height,
            channels=3,
            original_mode=original_mode,
            format_name=format_name,
            file_size_bytes=len(raw_bytes),
            exif_transposed=exif_transposed,
            has_transparency=has_transparency,
            warnings=warnings,
            error_message=dim_err,
        )

    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    return ImageLoadResult(
        is_valid=True,
        rgb_image=rgb_arr,
        bgr_image=bgr_arr,
        alpha_mask=alpha_mask,
        width=width,
        height=height,
        channels=3,
        original_mode=original_mode,
        format_name=format_name,
        file_size_bytes=len(raw_bytes),
        exif_transposed=exif_transposed,
        has_transparency=has_transparency,
        filename=filename,
        warnings=warnings,
        error_message=None,
    )


def _convert_pil_to_rgb(
    pil_img: Image.Image,
) -> Tuple[np.ndarray, Optional[np.ndarray], bool, List[str]]:
    """Convert any PIL image mode to a clean uint8 RGB array while preserving alpha mask."""
    warnings: List[str] = []
    alpha_mask: Optional[np.ndarray] = None
    has_transparency = False
    mode = pil_img.mode

    if mode in ("RGBA", "LA") or (mode == "P" and "transparency" in pil_img.info):
        rgba_img = pil_img.convert("RGBA")
        rgba_arr = np.array(rgba_img, dtype=np.uint8)
        alpha_channel = rgba_arr[:, :, 3]
        if np.any(alpha_channel < 255):
            has_transparency = True
            alpha_mask = alpha_channel.copy()
            warnings.append(
                "Input image contains transparency (alpha channel). Transparent regions "
                "were recorded in the alpha mask and composited over a neutral dark background."
            )
            # Composite onto neutral dark gray (30, 30, 30) to aid foreground segmentation
            alpha_f = (alpha_channel.astype(np.float32) / 255.0)[:, :, np.newaxis]
            bg = np.full_like(rgba_arr[:, :, :3], 30, dtype=np.float32)
            comp = rgba_arr[:, :, :3].astype(np.float32) * alpha_f + bg * (1.0 - alpha_f)
            rgb_arr = np.clip(comp, 0, 255).astype(np.uint8)
        else:
            rgb_arr = rgba_arr[:, :, :3].copy()
    elif mode in ("L", "I", "F", "1"):
        warnings.append(
            f"Input image is single-channel grayscale (mode='{mode}'). "
            "Color and stain analysis will operate with reduced chromatic information."
        )
        gray_arr = np.array(pil_img.convert("L"), dtype=np.uint8)
        rgb_arr = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2RGB)
    elif mode == "RGB":
        rgb_arr = np.array(pil_img, dtype=np.uint8)
    else:
        rgb_arr = np.array(pil_img.convert("RGB"), dtype=np.uint8)

    return rgb_arr, alpha_mask, has_transparency, warnings


def _validate_numpy_image(
    arr: np.ndarray, input_color_order: str = "RGB"
) -> ImageLoadResult:
    """Validate and normalize a direct NumPy array image."""
    warnings: List[str] = []
    if arr is None or not isinstance(arr, np.ndarray) or arr.size == 0:
        return ImageLoadResult(
            is_valid=False,
            error_message="Input NumPy image array is empty or None.",
        )

    if arr.ndim not in (2, 3):
        return ImageLoadResult(
            is_valid=False,
            error_message=f"Unsupported array dimensions (ndim={arr.ndim}). Expected 2D or 3D image.",
        )

    if not np.all(np.isfinite(arr)):
        return ImageLoadResult(
            is_valid=False,
            error_message="Image array contains NaN or infinite values.",
        )

    # Normalize dtype to uint8 safely
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(np.max(arr)) if arr.size > 0 else 0.0
            if max_val <= 1.0 and float(np.min(arr)) >= 0.0:
                arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    alpha_mask: Optional[np.ndarray] = None
    has_transparency = False

    if arr.ndim == 2:
        original_mode = "L"
        warnings.append(
            "Input array is single-channel grayscale. Chromatic stain analysis will be limited."
        )
        rgb_arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 1:
        original_mode = "L"
        warnings.append(
            "Input array is single-channel grayscale. Chromatic stain analysis will be limited."
        )
        rgb_arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif arr.shape[2] == 3:
        original_mode = input_color_order.upper()
        if original_mode == "BGR":
            rgb_arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            rgb_arr = arr.copy()
    elif arr.shape[2] == 4:
        original_mode = "RGBA" if input_color_order.upper() == "RGB" else "BGRA"
        alpha_channel = arr[:, :, 3]
        if np.any(alpha_channel < 255):
            has_transparency = True
            alpha_mask = alpha_channel.copy()
            warnings.append("Input 4-channel array contains transparent pixels.")
        color_part = arr[:, :, :3]
        if input_color_order.upper() == "BGR":
            color_part = cv2.cvtColor(color_part, cv2.COLOR_BGR2RGB)
        if has_transparency:
            alpha_f = (alpha_channel.astype(np.float32) / 255.0)[:, :, np.newaxis]
            bg = np.full_like(color_part, 30, dtype=np.float32)
            rgb_arr = np.clip(
                color_part.astype(np.float32) * alpha_f + bg * (1.0 - alpha_f),
                0,
                255,
            ).astype(np.uint8)
        else:
            rgb_arr = color_part.copy()
    else:
        return ImageLoadResult(
            is_valid=False,
            error_message=f"Unsupported channel count: {arr.shape[2]}.",
        )

    height, width = rgb_arr.shape[:2]
    dim_ok, dim_err, dim_warns = _check_dimensions_and_content(rgb_arr)
    warnings.extend(dim_warns)

    if not dim_ok:
        return ImageLoadResult(
            is_valid=False,
            width=width,
            height=height,
            channels=3,
            original_mode=original_mode,
            has_transparency=has_transparency,
            warnings=warnings,
            error_message=dim_err,
        )

    bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    return ImageLoadResult(
        is_valid=True,
        rgb_image=rgb_arr,
        bgr_image=bgr_arr,
        alpha_mask=alpha_mask,
        width=width,
        height=height,
        channels=3,
        original_mode=original_mode,
        format_name="NUMPY",
        file_size_bytes=int(arr.nbytes),
        has_transparency=has_transparency,
        filename="numpy_array",
        warnings=warnings,
    )


def _check_dimensions_and_content(
    rgb_arr: np.ndarray,
) -> Tuple[bool, Optional[str], List[str]]:
    """Validate minimum dimensions and check for blank/degenerate images."""
    warnings: List[str] = []
    height, width = rgb_arr.shape[:2]

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        return (
            False,
            f"Image dimensions ({width}x{height} px) are too small for analysis. "
            f"Minimum required dimensions are {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT} px.",
            warnings,
        )

    if max(width, height) < RECOMMENDED_MIN_WIDTH or min(width, height) < RECOMMENDED_MIN_HEIGHT:
        warnings.append(
            f"Low image resolution ({width}x{height} px). Fine fold lines and subtle "
            f"texture wear are best analyzed at >= {RECOMMENDED_MIN_WIDTH}x{RECOMMENDED_MIN_HEIGHT} px."
        )

    # Check if image is completely uniform (e.g., all black or all white)
    std_val = float(np.std(rgb_arr))
    if std_val < 1.0:
        return (
            False,
            "Image contains virtually no visual content (uniform/blank pixel values).",
            warnings,
        )

    return True, None, warnings
