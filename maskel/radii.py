"""Radii of skeleton voxels: distance to the nearest background voxel, via a k-d tree."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def skeleton_radii(
    binary: np.ndarray,
    coords: np.ndarray,
    spacing: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Distance from each voxel in *coords* to the nearest background voxel of *binary*."""
    foreground = binary.astype(bool)
    # The nearest background voxel always has a face neighbour in the
    # foreground, so the background shell of the mask is all the tree needs.
    shell = ndi.binary_dilation(foreground) & ~foreground
    if len(coords) == 0 or not shell.any():
        return np.zeros(len(coords))
    step = np.ones(binary.ndim) if spacing is None else np.asarray(spacing, float)
    distances, _ = cKDTree(np.argwhere(shell) * step).query(coords * step)
    return distances
