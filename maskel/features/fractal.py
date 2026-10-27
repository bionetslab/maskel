"""Box-counting fractal dimension of a skeleton."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from maskel._utils import to_binary


@njit(parallel=True, cache=True)
def _box_count_2d(img: np.ndarray, scale: int) -> int:
    """Count occupied boxes of size `scale` in a 2D binary image."""
    H, W = img.shape
    nh = (H + scale - 1) // scale
    nw = (W + scale - 1) // scale
    count = 0
    for i in prange(nh):
        for j in range(nw):
            r0 = i * scale
            c0 = j * scale
            r1 = min(r0 + scale, H)
            c1 = min(c0 + scale, W)
            found = False
            for r in range(r0, r1):
                if found:
                    break
                for c in range(c0, c1):
                    if img[r, c]:
                        found = True
                        break
            if found:
                count += 1
    return count


@njit(parallel=True, cache=True)
def _box_count_3d(img: np.ndarray, scale: int) -> int:
    """Count occupied boxes of size `scale` in a 3D binary volume."""
    D, H, W = img.shape
    nd = (D + scale - 1) // scale
    nh = (H + scale - 1) // scale
    nw = (W + scale - 1) // scale
    count = 0
    for i in prange(nd):
        for j in range(nh):
            for k in range(nw):
                z0 = i * scale
                y0 = j * scale
                x0 = k * scale
                z1 = min(z0 + scale, D)
                y1 = min(y0 + scale, H)
                x1 = min(x0 + scale, W)
                found = False
                for z in range(z0, z1):
                    if found:
                        break
                    for y in range(y0, y1):
                        if found:
                            break
                        for x in range(x0, x1):
                            if img[z, y, x]:
                                found = True
                                break
                if found:
                    count += 1
    return count


def fractal_dimension(skeleton: np.ndarray) -> tuple[float, float]:
    """Estimate fractal dimension of a binary skeleton via box-counting.

    Returns
    -------
    fd : float
        Fractal dimension (slope magnitude of log-log fit).
    r2 : float
        R^2 of the log-log linear fit (fit quality indicator).
    """
    binary = to_binary(skeleton)
    min_side = min(binary.shape)
    scale_base = min_side // 4
    if scale_base < 1:
        return 0.0, 0.0
    max_exp = int(np.floor(np.log2(scale_base)))
    if max_exp < 1:
        return 0.0, 0.0

    scales = np.array([2**k for k in range(1, max_exp + 1)], dtype=np.int64)

    box_counter = _box_count_2d if binary.ndim == 2 else _box_count_3d
    counts = np.array([box_counter(binary, int(s)) for s in scales], dtype=np.float64)

    # scales with zero count collapses the log
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0, 0.0

    log_s = np.log(scales[valid].astype(np.float64))
    log_n = np.log(counts[valid])

    coeffs = np.polyfit(log_s, log_n, 1)
    fd = float(-coeffs[0])

    predicted = np.polyval(coeffs, log_s)
    ss_res = float(np.sum((log_n - predicted) ** 2))
    ss_tot = float(np.sum((log_n - log_n.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return fd, float(r2)
