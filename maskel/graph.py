"""Skeleton voxel coordinates -> skan graph, without building an image."""

from __future__ import annotations

import itertools

import numpy as np
from scipy import sparse
from skan import Skeleton
from skan.csr import PathGraph, _mst_junctions


def build_graph(
    coords: np.ndarray,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None = None,
) -> Skeleton | None:
    """skan graph of the voxels *coords* (sorted by position, so node i is coords[i]); None without edges."""
    ndim = len(shape)
    # Stored as skan.Skeleton stores it: the dtype carries through to
    # summarize's coord columns (integer for the default unit spacing).
    spacing_arr = np.full(ndim, 1) if spacing is None else np.asarray(spacing)
    step = np.ones(ndim) * spacing_arr
    coords = np.asarray(coords, dtype=np.intp).reshape(-1, ndim)
    if len(coords) < 2:
        return None
    linear = np.ravel_multi_index(coords.T, shape)
    order = np.argsort(linear)
    linear, coords = linear[order], coords[order]

    # The pixel graph skan.csr.pixel_graph would build from an image:
    # every pair of 26-adjacent voxels, weighted by their physical distance.
    rows, cols, lengths = [], [], []
    for offset in itertools.product((-1, 0, 1), repeat=ndim):
        if not any(offset):
            continue
        neighbors = coords + offset
        inside = np.all((neighbors >= 0) & (neighbors < shape), axis=1)
        neighbor_linear = np.ravel_multi_index(neighbors[inside].T, shape)
        found = np.minimum(np.searchsorted(linear, neighbor_linear), linear.size - 1)
        hit = linear[found] == neighbor_linear
        rows.append(np.flatnonzero(inside)[hit])
        cols.append(found[hit])
        length = np.sqrt(np.sum((np.asarray(offset) * step) ** 2))
        lengths.append(np.full(int(hit.sum()), length))

    n = linear.size
    adjacency = sparse.coo_matrix(
        (np.concatenate(lengths), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    if adjacency.nnz == 0:
        return None
    # skan's next step after the pixel graph: clusters of adjacent junction
    # voxels are replaced by their minimum spanning tree.
    adjacency = _mst_junctions(adjacency)

    path_graph = PathGraph.from_graph(
        adj=adjacency,
        node_coordinates=coords,
        node_values=np.ones(n, dtype=np.float64),
        spacing=spacing_arr,
    )
    graph = Skeleton.from_path_graph(path_graph)
    graph.skeleton_shape = tuple(shape)
    graph.skeleton_dtype = np.dtype(np.uint8)
    return graph
