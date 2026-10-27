"""Local radii from the mask's distance transform, per point and per branch."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from skan import Skeleton


def compute_radii(
    binary: np.ndarray,
    skeleton: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    *,
    edt: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute local radii via Euclidean distance transform of the binary mask.

    The distance transform gives each foreground pixel its distance to the
    nearest background pixel. Sampling these values at skeleton positions
    yields the local radius at every centerline point.

    Parameters
    ----------
    binary : ndarray
        Binary mask (foreground > 0).
    skeleton : ndarray
        Binary skeleton of the same shape.
    spacing : tuple[float, ...], optional
        Per-axis physical size of one pixel/voxel. ``None`` (the default)
        keeps isotropic unit spacing, matching scipy's own default.
    edt : ndarray, optional
        Pre-computed Euclidean distance transform of *binary* (same shape),
        for a caller that already has one from an earlier call on this same
        *binary* - e.g. the pipeline calling this once before junction
        cleanup and again afterwards for the final radius matrix, where only
        the skeleton positions being sampled change, not the underlying
        mask. Computed internally via ``distance_transform_edt`` when not
        given.

    Returns
    -------
    radius_matrix : ndarray, shape=binary.shape, dtype=float64
        Array with radius values at skeleton pixels and zero elsewhere.
    stats : dict[str, float]
        Aggregated statistics (keys: ``mean_radius``, ``std_radius``,
        ``min_radius``, ``max_radius``, ``mean_diameter``, ``std_diameter``,
        ``min_diameter``, ``max_diameter``).
    """
    if edt is None:
        edt = distance_transform_edt(binary, sampling=spacing)
    radius_matrix = np.zeros_like(binary, dtype=np.float64)
    skeleton_mask = skeleton > 0
    radius_matrix[skeleton_mask] = edt[skeleton_mask]

    radii = radius_matrix[radius_matrix > 0]
    if radii.size:
        mean_r = float(np.mean(radii))
        std_r = float(np.std(radii))
        min_r = float(np.min(radii))
        max_r = float(np.max(radii))
    else:
        mean_r = std_r = min_r = max_r = 0.0

    return radius_matrix, {
        "mean_radius": mean_r,
        "std_radius": std_r,
        "min_radius": min_r,
        "max_radius": max_r,
        "mean_diameter": 2.0 * mean_r,
        "std_diameter": 2.0 * std_r,
        "min_diameter": 2.0 * min_r,
        "max_diameter": 2.0 * max_r,
    }


def per_segment_radii(
    radius_matrix: np.ndarray,
    graph: Skeleton,
    n_branches: int,
    spacing: tuple[float, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-segment radius/diameter statistics from an EDT radius matrix.

    Parameters
    ----------
    radius_matrix : ndarray
        Array with radius values at skeleton pixels, zero elsewhere.
    graph : Skeleton
        Pre-built skan Skeleton graph.
    n_branches : int
        Number of branches (e.g. ``len(branch_data)``).
    spacing : tuple[float, ...], optional
        Per-axis physical size of one pixel/voxel, used to weight each
        step along a branch's path by its real physical length rather
        than one unit per pixel. ``None`` (the default) is equivalent to
        isotropic unit spacing.

    Returns
    -------
    dict[str, ndarray]
        Arrays of length ``n_branches`` with keys ``mean_radius``,
        ``std_radius``, ``min_radius``, ``max_radius``, ``mean_diameter``,
        ``std_diameter``, ``min_diameter``, ``max_diameter``, ``volume``,
        ``surface_area``.
        Branches with no positive radius values yield ``nan``.

    Notes
    -----
    ``volume``/``surface_area`` model the branch as a chain of frustums
    between consecutive skeleton pixels along its path: each step's
    physical length (accounting for diagonal steps, and for anisotropic
    *spacing*) is weighted by the average of its two endpoints' radii,
    rather than treating every pixel as contributing one unit of length
    regardless of step direction.
    """
    n = n_branches
    mean_r = np.full(n, np.nan, dtype=np.float64)
    std_r = np.full(n, np.nan, dtype=np.float64)
    min_r = np.full(n, np.nan, dtype=np.float64)
    max_r = np.full(n, np.nan, dtype=np.float64)
    volume = np.full(n, np.nan, dtype=np.float64)
    surface_area = np.full(n, np.nan, dtype=np.float64)

    spacing_arr = np.asarray(spacing, dtype=np.float64) if spacing is not None else None

    for i in range(n):
        coords = graph.path_coordinates(i)
        radii = radius_matrix[tuple(coords.T)]
        radii_valid = radii[radii > 0]
        if radii_valid.size:
            mean_r[i] = np.mean(radii_valid)
            std_r[i] = np.std(radii_valid)
            min_r[i] = np.min(radii_valid)
            max_r[i] = np.max(radii_valid)

            # Model the branch as a chain of frustums between consecutive
            # path points: each step's physical length (diagonal-aware,
            # and spacing-aware for anisotropic voxels) is weighted by the
            # average of its two endpoints' radii.
            steps = np.diff(coords, axis=0).astype(np.float64)
            if spacing_arr is not None:
                steps = steps * spacing_arr
            step_lengths = np.linalg.norm(steps, axis=1)
            r_mid = (radii[:-1] + radii[1:]) / 2.0

            volume[i] = np.pi * float(np.sum(r_mid**2 * step_lengths))
            surface_area[i] = 2.0 * np.pi * float(np.sum(r_mid * step_lengths))

    return {
        "mean_radius": mean_r,
        "std_radius": std_r,
        "min_radius": min_r,
        "max_radius": max_r,
        "mean_diameter": 2.0 * mean_r,
        "std_diameter": 2.0 * std_r,
        "min_diameter": 2.0 * min_r,
        "max_diameter": 2.0 * max_r,
        "volume": volume,
        "surface_area": surface_area,
    }


def aggregate_segment_stats(branch_data) -> dict[str, float]:
    """Object-level aggregates of the per-branch ``volume``/``surface_area``
    columns produced by `per_segment_radii` (mean across this object's
    branches).

    Requires *branch_data* to already carry those two columns (only true
    after `per_segment_radii`'s output has been merged into it) and to be
    non-empty - a caller that hasn't done that, or has no branches, should
    not call this.
    """
    return {
        "mean_segment_volume": float(np.nanmean(branch_data["volume"])),
        "mean_surface_area": float(np.nanmean(branch_data["surface_area"])),
    }
