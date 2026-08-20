"""Lee94 thinning algorithm for 3D binary images.

Originally derived from scikit-image, BSD-3-Clause
https://github.com/scikit-image/scikit-image

This is a pure Python implementation of the
thinning algorithm from [Lee94], based on the scikit-image Cython
implementation in `skimage.morphology._skeletonize_lee_cy` (`_compute_thin_image`)
[SKIMAGE], which itself is a port of the Skeletonize3D ImageJ plugin by
Ignacio Arganda-Carreras [IAC15].

References
----------
- [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
          via 3-D medial surface/axis thinning algorithms.
          Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.

- [IAC15] Ignacio Arganda-Carreras, 2015. Skeletonize3D plugin for ImageJ(C).
           https://imagej.net/Skeletonize3D

- [SKIMAGE] scikit-image, `skimage.morphology._skeletonize_lee_cy`.
           https://github.com/scikit-image/scikit-image/blob/main/src/skimage/morphology/_skeletonize_lee_cy.pyx.in
"""

import numpy as np
from numba import njit, prange

# _compute_thin_image stores candidate voxel coordinates as uint16 to keep its
# scratch buffers small.  Those coordinates index the one-voxel-padded volume
# and reach ``dim`` for an input axis of length ``dim``, so every axis has to
# fit in a uint16.  Numba would wrap silently past that, hence the explicit
# check in thin_3d.
_MAX_DIM = int(np.iinfo(np.uint16).max)

# ---------------------------------------------------------------------------
# Lookup tables for Euler characteristic computation (Lee94 Table 2).
# _EULER_ARR maps the 256 possible 3x3x3 configurations to their Euler
# delta values.  Only odd indices (center=1) are populated; evens are 0.
# ---------------------------------------------------------------------------
_EULER_ARR = np.array(
    [
        1,
        -1,
        -1,
        1,
        -3,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        -3,
        -1,
        3,
        1,
        1,
        -1,
        3,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        -3,
        3,
        -1,
        1,
        1,
        3,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        1,
        3,
        3,
        1,
        5,
        3,
        3,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        -7,
        -1,
        -1,
        1,
        -3,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        -3,
        -1,
        3,
        1,
        1,
        -1,
        3,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        -3,
        3,
        -1,
        1,
        1,
        3,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
        1,
        3,
        3,
        1,
        5,
        3,
        3,
        1,
        -1,
        1,
        1,
        -1,
        3,
        1,
        1,
        -1,
    ],
    dtype=np.int32,
)

_EULER_LUT = np.zeros(256, dtype=np.int32)
_EULER_LUT[1::2] = _EULER_ARR

# ---------------------------------------------------------------------------
# Octant masks for Euler-invariant check.
# Each row lists 7 neighbour indices (in the flat 27-element neighbourhood)
# that form one octant of the 3x3x3 cube. The octant's bits are packed into
# a number used to index _EULER_LUT.
# ---------------------------------------------------------------------------
_OCTANTS = np.array(
    [
        [2, 1, 11, 10, 5, 4, 14],
        [0, 9, 3, 12, 1, 10, 4],
        [8, 7, 17, 16, 5, 4, 14],
        [6, 15, 7, 16, 3, 12, 4],
        [20, 23, 19, 22, 11, 14, 10],
        [18, 21, 9, 12, 19, 22, 10],
        [26, 23, 17, 14, 25, 22, 16],
        [24, 25, 15, 16, 21, 22, 12],
    ],
    dtype=np.int64,
)

# Six border directions processed in order: -z, z, -y, y, -x, x.
_BORDERS = np.array([4, 3, 2, 1, 5, 6], dtype=np.int64)

# Offsets for all 26 neighbours of a voxel (excluding the centre).
_OFFSETS_26 = np.array(
    [
        (-1, -1, -1),
        (-1, -1, 0),
        (-1, -1, 1),
        (-1, 0, -1),
        (-1, 0, 0),
        (-1, 0, 1),
        (-1, 1, -1),
        (-1, 1, 0),
        (-1, 1, 1),
        (0, -1, -1),
        (0, -1, 0),
        (0, -1, 1),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
        (1, -1, -1),
        (1, -1, 0),
        (1, -1, 1),
        (1, 0, -1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, -1),
        (1, 1, 0),
        (1, 1, 1),
    ],
    dtype=np.int8,
)

# ---------------------------------------------------------------------------
# Pre-computed 26-neighbour adjacency graph.
# _ADJ26[i, j] == 1  iff voxels i and j (in the 26-neighbour set) are
#    themselves 26-adjacent.
# _ADJ26_LIST[i, :k]  holds the list of adjacency neighbours for voxel i.
# _ADJ26_COUNT[i]     is the number of such neighbours (= k).
#
# These are used by _is_simple_point to run DFS on the 26-neighbour graph.
# ---------------------------------------------------------------------------
_ADJ26 = np.zeros((26, 26), dtype=np.uint8)
for _i in range(26):
    for _j in range(26):
        if _i == _j:
            continue
        dp = int(_OFFSETS_26[_i, 0]) - int(_OFFSETS_26[_j, 0])
        dr = int(_OFFSETS_26[_i, 1]) - int(_OFFSETS_26[_j, 1])
        dc = int(_OFFSETS_26[_i, 2]) - int(_OFFSETS_26[_j, 2])
        if abs(dp) <= 1 and abs(dr) <= 1 and abs(dc) <= 1:
            _ADJ26[_i, _j] = 1

_ADJ26_LIST = np.full((26, 26), -1, dtype=np.int8)
_ADJ26_COUNT = np.zeros(26, dtype=np.uint8)
for _i in range(26):
    _count = 0
    for _j in range(26):
        if _ADJ26[_i, _j] == 1:
            _ADJ26_LIST[_i, _count] = _j
            _count += 1
    _ADJ26_COUNT[_i] = _count

# Offsets for the six face-neighbour directions (and identity at index 0).
_BORDER_OFFSETS = np.array(
    [
        (0, 0, 0),
        (0, 0, -1),
        (0, 0, 1),
        (0, 1, 0),
        (0, -1, 0),
        (1, 0, 0),
        (-1, 0, 0),
    ],
    dtype=np.int8,
)


# ======================== Low-level predicates ===========================


@njit(cache=True)
def _get_neighborhood(img, p, r, c, neighborhood):
    """Fill ``neighborhood`` with the 27 voxels of the 3×3×3 cube at (p,r,c)."""
    idx = 0
    for dp in range(-1, 2):
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                neighborhood[idx] = img[p + dp, r + dr, c + dc]
                idx += 1


@njit(cache=True)
def _is_endpoint(neighbors):
    """A voxel is an endpoint if exactly 2 of its 27 neighbours are foreground."""
    s = 0
    for j in range(27):
        s += neighbors[j]
    return s == 2


@njit(cache=True)
def _is_euler_invariant(neighbors):
    """Return True if removing the centre voxel preserves the Euler
    characteristic.  Packs each octant into a bit mask, then looks up the
    Euler delta from _EULER_LUT."""
    euler_char = 0
    for octant in range(8):
        n = 1
        for j in range(7):
            idx = _OCTANTS[octant, j]
            if neighbors[idx] == 1:
                n |= 1 << (7 - j)
        euler_char += _EULER_LUT[n]
    return euler_char == 0


@njit(cache=True)
def _is_simple_point(neighbors, cube, visited, stack):
    """Return True if the centre voxel is a *simple point* - i.e. removing it
    does not change the topology of the foreground.

    Works by extracting the 26 exterior voxels into ``cube``, then counting
    connected components on the 26-adjacency graph via DFS.  If there is
    exactly one connected component among the foreground neighbours, the
    point is simple.

    ``cube``, ``visited``, ``stack`` are pre-allocated scratch buffers."""
    j = 0
    for i in range(27):
        if i == 13:
            continue
        cube[j] = neighbors[i]
        j += 1

    visited[:] = 0
    components = 0

    for i in range(26):
        if cube[i] != 1 or visited[i] == 1:
            continue

        components += 1
        if components >= 2:
            return False

        sp = 0
        stack[sp] = i
        sp += 1
        visited[i] = 1

        while sp > 0:
            sp -= 1
            cur = stack[sp]
            for k in range(_ADJ26_COUNT[cur]):
                nxt = _ADJ26_LIST[cur, k]
                if cube[nxt] != 1 or visited[nxt] == 1:
                    continue
                visited[nxt] = 1
                stack[sp] = nxt
                sp += 1

    return True


# ====================== Candidate finding and marking =====================


@njit(cache=True, parallel=True)
def _find_simple_point_candidates(img, curr_border, candidates):
    """Scan the volume for foreground voxels on the face given by
    ``_BORDER_OFFSETS[curr_border]`` and write their coordinates into
    ``candidates``.

    Two-pass parallel strategy (avoids a shared counter across threads):
      1. Each z-slice counts its candidates in parallel.
      2. A prefix sum computes write offsets.
      3. Each z-slice fills its contiguous segment of ``candidates``.
    """
    dp = int(_BORDER_OFFSETS[curr_border, 0])
    dr = int(_BORDER_OFFSETS[curr_border, 1])
    dc = int(_BORDER_OFFSETS[curr_border, 2])

    P = img.shape[0] - 1
    R = img.shape[1] - 1
    C = img.shape[2] - 1
    num_slices = P - 1

    slice_counts = np.zeros(num_slices, dtype=np.int64)
    for p in prange(1, P):
        local_count = 0
        for r in range(1, R):
            for c in range(1, C):
                if img[p, r, c] == 1 and img[p + dp, r + dr, c + dc] == 0:
                    local_count += 1
        slice_counts[p - 1] = local_count

    offsets = np.zeros(num_slices + 1, dtype=np.int64)
    for p_idx in range(num_slices):
        offsets[p_idx + 1] = offsets[p_idx] + slice_counts[p_idx]
    total = offsets[num_slices]

    for p in prange(1, P):
        idx = offsets[p - 1]
        for r in range(1, R):
            for c in range(1, C):
                if img[p, r, c] == 1 and img[p + dp, r + dr, c + dc] == 0:
                    candidates[idx, 0] = p
                    candidates[idx, 1] = r
                    candidates[idx, 2] = c
                    idx += 1

    return total


@njit(cache=True, parallel=True)
def _mark_removable_candidates(img, candidates, num_candidates, removable):
    """For each candidate voxel, check the three Lee94 criteria in order:
      1. not an endpoint,
      2. Euler-characteristic invariant,
      3. simple point.

    Voxels that pass all three are marked as removable (1).  The checks
    short-circuit: failure on any earlier criterion skips the later ones.
    """
    for i in prange(num_candidates):
        p = candidates[i, 0]
        r = candidates[i, 1]
        c = candidates[i, 2]

        neighborhood = np.empty(27, dtype=np.uint8)
        _get_neighborhood(img, p, r, c, neighborhood)

        cube = np.empty(26, dtype=np.uint8)
        visited = np.zeros(26, dtype=np.uint8)
        stack = np.empty(26, dtype=np.int64)

        can_remove = (
            (not _is_endpoint(neighborhood))
            and _is_euler_invariant(neighborhood)
            and _is_simple_point(neighborhood, cube, visited, stack)
        )
        removable[i] = 1 if can_remove else 0


# ======================== Sequential removal pass =========================


@njit(cache=True)
def _apply_removals(img, candidates, num_candidates, removable, removed_epoch, tag):
    """Sequentially remove voxels marked as removable, re-checking simplicity
    only when a neighbour was *already removed in the same batch*.

    The naive approach would re-read the neighbourhood and re-run
    _is_simple_point for *every* candidate (which is expensive).  However,
    _mark_removable_candidates already verified that each candidate is simple
    *before any removals in this batch*.  If none of a candidate's 26
    neighbours have been removed yet in this batch, the old verification
    is still valid and we can skip the re-check.

    ``removed_epoch`` is a 3-D stamp array - ``tag`` (a value in 1..255) is
    written at the position of every voxel removed in this call.  Checking
    whether a neighbour was removed is an O(26) stamp lookup, not an O(k) scan
    over previously removed coordinates.
    """
    removed = 0
    neighborhood = np.empty(27, dtype=np.uint8)
    cube = np.empty(26, dtype=np.uint8)
    visited = np.zeros(26, dtype=np.uint8)
    stack = np.empty(26, dtype=np.int64)
    for i in range(num_candidates):
        if removable[i] == 0:
            continue
        p = candidates[i, 0]
        r = candidates[i, 1]
        c = candidates[i, 2]
        if img[p, r, c] != 1:
            continue

        neighbor_removed = False
        for dp in range(-1, 2):
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dp == 0 and dr == 0 and dc == 0:
                        continue
                    if removed_epoch[p + dp, r + dr, c + dc] == tag:
                        neighbor_removed = True
                        break
                if neighbor_removed:
                    break
            if neighbor_removed:
                break

        if neighbor_removed:
            _get_neighborhood(img, p, r, c, neighborhood)
            if not _is_simple_point(neighborhood, cube, visited, stack):
                continue

        img[p, r, c] = 0
        removed_epoch[p, r, c] = tag
        removed += 1
    return removed


@njit(cache=True)
def _compute_thin_image(img):
    """Iteratively peel surface voxels from a padded 3D binary volume until
    a 1-voxel-thin skeleton remains.

    The outer loop processes all six face directions sequentially.  A border
    direction is considered "stable" when no removable voxel was found on
    that face during the pass.  When all six are stable the skeleton is done.
    """
    num_borders = 6
    unchanged_borders = 0

    # Use #fg voxels as upper bound for number of candidates
    num_foreground = np.count_nonzero(img)
    candidates = np.empty((num_foreground, 3), dtype=np.uint16)
    removable = np.empty(num_foreground, dtype=np.uint8)
    # uint8's tradeoff: less ram usage vs. memset removed_epoch[:] = 0 every 255 epochs.
    removed_epoch = np.zeros(img.shape, dtype=np.uint8)
    tag = 0

    while unchanged_borders < num_borders:
        unchanged_borders = 0
        for j in range(num_borders):
            curr_border = _BORDERS[j]
            num_candidates = _find_simple_point_candidates(img, curr_border, candidates)

            if num_candidates == 0:
                unchanged_borders += 1
                continue

            _mark_removable_candidates(img, candidates, num_candidates, removable)

            tag += 1
            if tag > 255:
                # All of 1..255 have been used; reuse would alias live stamps.
                removed_epoch[:] = 0
                tag = 1

            removed = _apply_removals(
                img,
                candidates,
                num_candidates,
                removable,
                removed_epoch,
                tag,
            )

            if removed == 0:
                unchanged_borders += 1

    return img


def thin_3d(img):
    """Lee94 thinning algorithm for a 3D binary volume.

    Parameters
    ----------
    img : ndarray
        3D binary volume (0=background, 1=foreground).

    Returns
    -------
    ndarray
        Thinned binary volume with the same shape as img.

    Raises
    ------
    ValueError
        If input is not 3-dimensional, or if any axis is longer than
        ``_MAX_DIM`` voxels.
    """
    if img.ndim != 3:
        raise ValueError(f"Expected 3D input, got {img.ndim}D")
    if max(img.shape) > _MAX_DIM:
        raise ValueError(
            f"Volume too large for 3D thinning: shape={img.shape}. "
            f"Each axis must be at most {_MAX_DIM} voxels "
            "(voxel coordinates are stored as uint16)."
        )

    padded = np.zeros(
        (img.shape[0] + 2, img.shape[1] + 2, img.shape[2] + 2),
        dtype=np.uint8,
    )
    padded[1:-1, 1:-1, 1:-1] = img

    out = _compute_thin_image(padded)
    return out[1:-1, 1:-1, 1:-1].copy()
