"""Box-counting fractal dimension of a skeleton."""

from __future__ import annotations

import numpy as np


def fractal_dimension(coords: np.ndarray) -> tuple[float, float]:
    """Box-counting dimension and fit R^2 of the voxels *coords*, boxes anchored at their bounding box."""
    if len(coords) == 0:
        return 0.0, 0.0
    origin = coords.min(axis=0)
    min_side = int((coords.max(axis=0) - origin + 1).min())
    if min_side < 16:  # fewer than two box sizes to fit a slope to
        return 0.0, 0.0
    max_exp = int(np.floor(np.log2(min_side // 4)))
    scales = 2 ** np.arange(1, max_exp + 1)
    counts = np.array([_occupied_boxes(coords - origin, s) for s in scales], float)

    log_s = np.log(scales.astype(float))
    log_n = np.log(counts)
    coeffs = np.polyfit(log_s, log_n, 1)
    predicted = np.polyval(coeffs, log_s)
    ss_res = float(np.sum((log_n - predicted) ** 2))
    ss_tot = float(np.sum((log_n - log_n.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return float(-coeffs[0]), float(r2)


def _occupied_boxes(coords: np.ndarray, size: int) -> int:
    """Number of boxes of *size* voxels per side that contain any of *coords*."""
    boxes = coords // size
    return len(np.unique(np.ravel_multi_index(boxes.T, boxes.max(axis=0) + 1)))
