"""Morphological preprocessing of the binary mask before thinning:
opening, closing and hole filling.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from maskel._utils import _label_compact
from maskel.config import PipelineConfig


def preprocess_binary(
    binary: np.ndarray,
    closing_iterations: int = 0,
    fill_holes: bool = False,
    max_hole_size: int = 0,
    opening_radius: int = 0,
) -> np.ndarray:
    """Apply morphological preprocessing to a binary mask.

    Steps run in this order: opening, closing, hole filling.

    Parameters
    ----------
    binary : ndarray
        Binary uint8 array (0/1).
    closing_iterations : int
        Radius of the ball (3D) / disk (2D) used for binary closing, in
        voxels (0 = skip). Away from the array border this gives the same
        result as ``skimage.morphology.closing(binary, ball(r))``; at the
        border, edge voxels are repeated outward first (see
        ``_ball_morphology``). A Euclidean ball
        rather than a cross applied *r* times, which would only reach about
        r/sqrt(3) along diagonals. The name is kept for config
        compatibility; radius 1 equals one iteration of the 6/4-connected
        cross.
    fill_holes : bool
        Whether to fill enclosed background regions.
    max_hole_size : int
        Max hole size in voxels to fill (0 = unlimited).
    opening_radius : int
        Radius of the ball (3D) / disk (2D) used for binary opening, in
        voxels (0 = skip). Removes thin bridges, flaps and specks that the
        ball doesn't fit into. Runs before the closing; same border
        handling as the closing.

    Returns
    -------
    ndarray
        Preprocessed binary array, same dtype as *binary*.

    Notes
    -----
    Keeps a single working array through closing/fill-holes instead of
    round-tripping through *binary*'s original dtype after each step, and
    avoids ever materializing the extra full-size int8 copies the previous
    implementation used just to detect which voxels fill_holes changed
    (``result > before_fill`` gives the same boolean mask directly, and
    correctly - unlike ``~before_fill``, a bitwise-not on a non-bool array
    doesn't mean logical negation). ``before_fill`` needs no explicit
    ``.copy()`` either: ``binary_fill_holes`` never mutates its input in
    place, so simply not reassigning over the old reference until a new
    array exists already keeps the "before" snapshot alive.

    This matters beyond avoiding a few small temporaries: on a large 3D
    volume, connected-component labeling of the filled regions (used to
    enforce *max_hole_size*) can by itself need far more peak memory than
    any of maskel's own thinning methods do, since ``scipy.ndimage.label``
    defaults to a 32-bit label array (4 bytes/voxel). ``_label_compact``
    below tries a 16-bit array first, which is enough for realistic hole
    counts and halves that allocation.
    """
    if closing_iterations <= 0 and opening_radius <= 0 and not fill_holes:
        return binary

    dtype = binary.dtype
    result = binary

    if opening_radius > 0:
        result = _ball_morphology(result, opening_radius, "opening")

    if closing_iterations > 0:
        result = _ball_morphology(result, closing_iterations, "closing")

    if fill_holes:
        before_fill = result if max_hole_size > 0 else None
        result = ndi.binary_fill_holes(result)

        if max_hole_size > 0 and before_fill is not None:
            filled = result > before_fill
            del before_fill
            if filled.any():
                labels, _ = _label_compact(filled)
                del filled
                sizes = np.bincount(labels.ravel())
                big = sizes > max_hole_size
                big[0] = False
                result[big[labels]] = False

    return result.astype(dtype, copy=False)


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


def _preprocessing_enabled(config: PipelineConfig) -> bool:
    """True when any step of ``preprocess_binary`` is switched on."""
    e = config.extraction
    return e.opening_radius > 0 or e.closing_iterations > 0 or e.fill_holes


def _preprocessing_reach(config: PipelineConfig) -> int:
    """How far (in voxels) the enabled preprocessing steps reach: 2r for an
    opening or closing with a ball of radius r. Hole filling only fills
    enclosed background, so it never reaches outside. Neither opening nor
    closing grows an object past its bounding box, so this is a safety
    margin for the crop rather than room the result needs."""
    e = config.extraction
    return 2 * e.opening_radius + 2 * e.closing_iterations
