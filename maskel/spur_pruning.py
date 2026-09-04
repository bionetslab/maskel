"""Removal of short spur branches (thinning artifacts) from a skeleton."""

import numpy as np
from skan.csr import Skeleton


def prune_short_spurs(
    skeleton: np.ndarray,
    graph: Skeleton,
    branch_data,
    min_length: float = 10.0,
) -> np.ndarray:
    """Remove short spur branches from a vessel skeleton.

    A spur is a branch that connects an endpoint (degree 1) directly to a
    junction (degree > 1) and is shorter than `min_length` - usually a
    thinning artifact rather than a real vessel tip. Endpoint-endpoint
    branches (isolated segments with no junction) and junction-junction
    branches are left untouched, since neither matches that degree pattern.

    Parameters
    ----------
    skeleton : ndarray
        Binary skeleton array the graph was built from.
    graph : Skeleton
        Pre-built skan Skeleton graph (e.g. from `build_skeleton_graph`).
    branch_data : DataFrame
        Branch summary from `skan.summarize(graph, ...)`, in the same row
        order as `graph`'s branches.
    min_length : float, optional
        Branches shorter than this (``branch-distance``) qualify for
        removal - in pixel units, or physical units if `graph` was built
        with a `spacing`. Default is 10.0.

    Returns
    -------
    cleaned : ndarray
        Binary skeleton of the same shape with short spurs removed.
    """
    if branch_data.empty:
        return skeleton.copy().astype(np.uint8)

    degrees = graph.degrees
    src_nodes = branch_data["node-id-src"].to_numpy(dtype=np.int64)
    dst_nodes = branch_data["node-id-dst"].to_numpy(dtype=np.int64)
    lengths = branch_data["branch-distance"].to_numpy(dtype=float)

    src_degree = degrees[src_nodes]
    dst_degree = degrees[dst_nodes]

    is_spur = (
        ((src_degree == 1) & (dst_degree > 1)) | ((dst_degree == 1) & (src_degree > 1))
    ) & (lengths < min_length)

    if not is_spur.any():
        return skeleton.copy().astype(np.uint8)

    if is_spur.all():
        # every branch got pruned away - nothing left worth keeping.
        return np.zeros_like(skeleton, dtype=np.uint8)

    cleaned = skeleton.copy().astype(np.uint8)
    for i in np.flatnonzero(is_spur):
        # path[0] is the src-node pixel, path[-1] the dst-node pixel (skan
        # convention). Drop every pixel on the endpoint side, keeping the
        # one pixel closest to the junction so it stays connected to the
        # rest of the skeleton.
        path = graph.path_coordinates(i)
        drop = path[:-1] if src_degree[i] == 1 else path[1:]
        cleaned[tuple(drop.T)] = 0

    return cleaned
