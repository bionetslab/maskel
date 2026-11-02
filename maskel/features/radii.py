"""Radius statistics per skeleton and per branch."""

from __future__ import annotations

import numpy as np
from skan import Skeleton


def radius_stats(radius: np.ndarray) -> dict[str, float]:
    """Mean, std, min and max radius and diameter over the positive radii."""
    radius = radius[radius > 0]
    if radius.size:
        mean_r, std_r = float(np.mean(radius)), float(np.std(radius))
        min_r, max_r = float(np.min(radius)), float(np.max(radius))
    else:
        mean_r = std_r = min_r = max_r = 0.0
    return {
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
    graph: Skeleton,
    radius: np.ndarray,
    spacing: tuple[float, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Radius statistics, volume and surface area (as chained frustums) of every branch."""
    n = graph.n_paths
    mean_r, std_r, min_r, max_r, volume, surface_area = (
        np.full(n, np.nan) for _ in range(6)
    )
    step = None if spacing is None else np.asarray(spacing, dtype=np.float64)

    for i in range(n):
        path = graph.path(i)
        radii = radius[path]
        valid = radii[radii > 0]
        if not valid.size:
            continue
        mean_r[i], std_r[i] = np.mean(valid), np.std(valid)
        min_r[i], max_r[i] = np.min(valid), np.max(valid)

        steps = np.diff(graph.coordinates[path], axis=0).astype(np.float64)
        if step is not None:
            steps = steps * step
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
    """Mean branch volume and surface area over a non-empty branch table."""
    return {
        "mean_segment_volume": float(np.nanmean(branch_data["volume"])),
        "mean_surface_area": float(np.nanmean(branch_data["surface_area"])),
    }
