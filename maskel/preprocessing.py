"""Morphological preprocessing of a binary mask before thinning."""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from maskel._utils import _label_compact
from maskel.config import ExtractionConfig


def preprocess_binary(
    binary: np.ndarray,
    *,
    opening_radius: int = 0,
    closing_iterations: int = 0,
    fill_holes: bool = False,
    max_hole_diameter: float = 0.0,
    spacing: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Open, close and fill holes (fitting in a box of *max_hole_diameter*, 0 = any) in *binary*."""
    if not preprocessing_enabled(opening_radius, closing_iterations, fill_holes):
        return binary

    dtype = binary.dtype
    result = binary
    if opening_radius > 0:
        result = _ball_morphology(result, opening_radius, "opening")
    if closing_iterations > 0:
        result = _ball_morphology(result, closing_iterations, "closing")
    if fill_holes:
        filled = ndi.binary_fill_holes(result)
        if max_hole_diameter > 0:
            _keep_small_holes(filled, filled > result, max_hole_diameter, spacing)
        result = filled
    return result.astype(dtype, copy=False)


def _keep_small_holes(
    filled: np.ndarray,
    holes: np.ndarray,
    max_diameter: float,
    spacing: tuple[float, ...] | None,
) -> None:
    """Empty again every hole in *filled* whose extent exceeds *max_diameter* on any axis."""
    if not holes.any():
        return
    labels, _ = _label_compact(holes)
    step = np.ones(holes.ndim) if spacing is None else np.asarray(spacing)
    for i, box in enumerate(ndi.find_objects(labels), start=1):
        extent = np.array([s.stop - s.start for s in box]) * step
        if (extent > max_diameter).any():
            filled[box][labels[box] == i] = False


def _ball_morphology(binary: np.ndarray, radius: int, op: str) -> np.ndarray:
    """Binary opening or closing with a Euclidean ball of *radius*.

    The array is padded by *radius* with its own edge voxels first. Without
    padding, scipy treats everything outside the array as background, so
    the erosion eats up to *radius* voxels into anything touching the array
    border. Treating the outside as foreground instead (skimage's
    ``mode="ignore"``) doesn't fit either: objects are processed in tight
    per-object crops, and it would fill the strip between an object and its
    crop edge. Repeating the edge voxels does neither - a crop's own 1-voxel
    background margin stays background, and an object cut off by the real
    image border continues past it instead of ending there.
    """
    footprint = _ball_footprint(binary.ndim, radius)
    padded = np.pad(binary.astype(bool, copy=False), radius, mode="edge")
    if op == "opening":
        padded = ndi.binary_erosion(padded, structure=footprint)
        padded = ndi.binary_dilation(padded, structure=footprint)
    elif op == "closing":
        padded = ndi.binary_dilation(padded, structure=footprint)
        padded = ndi.binary_erosion(padded, structure=footprint)
    else:
        raise ValueError(f"op must be 'opening' or 'closing', got {op!r}")
    return padded[(slice(radius, -radius),) * binary.ndim]


def _ball_footprint(ndim: int, radius: int) -> np.ndarray:
    """Euclidean ball of *radius* in *ndim* dimensions - the same voxels as
    ``skimage.morphology.ball`` (3D) / ``disk`` (2D), without needing
    scikit-image at runtime."""
    grid = np.ogrid[(slice(-radius, radius + 1),) * ndim]
    return sum(g**2 for g in grid) <= radius**2


def preprocessing_enabled(
    opening_radius: int, closing_iterations: int, fill_holes: bool
) -> bool:
    """True when any preprocessing step is switched on."""
    return opening_radius > 0 or closing_iterations > 0 or fill_holes


def crop_pad(extraction: ExtractionConfig) -> int:
    """Background voxels kept around each label's bounding box: 1 plus the ball steps' reach."""
    return 1 + 2 * extraction.opening_radius + 2 * extraction.closing_iterations


def preprocessing_margin(
    extraction: ExtractionConfig, spacing: tuple[float, ...] | None, ndim: int
) -> tuple[int, ...]:
    """Voxels per axis by which preprocessing near a block edge can differ from the whole image."""
    step = (1.0,) * ndim if spacing is None else spacing
    ball = 2 * extraction.opening_radius + 2 * extraction.closing_iterations
    if not (extraction.fill_holes and extraction.max_hole_diameter > 0):
        return (ball,) * ndim
    return tuple(ball + math.ceil(extraction.max_hole_diameter / s) + 1 for s in step)
