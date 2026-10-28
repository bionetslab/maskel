"""Skeleton -> skan graph, the basis of every branch and node feature."""

from __future__ import annotations

import itertools

import numpy as np
from scipy import sparse
from skan import Skeleton
from skan.csr import PathGraph, _mst_junctions

from maskel._utils import to_binary


def build_skeleton_graph(
    skeleton: np.ndarray, spacing: tuple[float, ...] | None = None
) -> Skeleton:
    """Build a graph representation from a binary skeleton.

    Parameters
    ----------
    skeleton : ndarray
        Binary skeleton array.
    spacing : tuple[float, ...], optional
        Per-axis physical size of one pixel/voxel, passed through to
        ``skan.Skeleton``. Once built with spacing, everything downstream
        via ``skan.summarize`` (``branch-distance``, ``euclidean-distance``,
        and therefore ``total_length``/``mean_length``/``tortuosity``/``hgu``)
        becomes physically correct automatically. ``graph.coordinates``
        stay raw pixel indices regardless of spacing. ``None`` (the
        default) keeps isotropic unit spacing.
    """
    return Skeleton(to_binary(skeleton), spacing=spacing if spacing is not None else 1)


def skeleton_graph_from_coordinates(
    coordinates: np.ndarray,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | np.ndarray | None = None,
) -> Skeleton | None:
    """Build the same graph as `build_skeleton_graph` from the skeleton's
    pixel coordinates alone, without allocating an image of *shape*.

    ``skan.Skeleton`` only needs the image to find the skeleton pixels and
    their neighbours; everything after that (junction-cluster simplification,
    path tracing, distances) runs on the sparse pixel graph. This builds that
    pixel graph from the coordinates instead - pixels numbered in raveled
    image order and edges weighted by their (spacing-scaled) length, exactly
    as ``skan.csr.pixel_graph`` does - and hands it to skan from there on, so
    node ids, paths and ``skan.summarize`` output match the image route.

    Parameters
    ----------
    coordinates : ndarray, shape (N, ndim)
        Integer pixel coordinates of the skeleton, in any order and without
        duplicates.
    shape : tuple[int, ...]
        Shape of the image the coordinates index into.
    spacing : tuple[float, ...] or ndarray, optional
        Per-axis physical size of one pixel/voxel, as for
        `build_skeleton_graph`.

    Returns
    -------
    Skeleton or None
        ``None`` when no two pixels are adjacent - skan cannot build a graph
        without edges (see ``pipeline._skeleton_has_no_branches``). The
        returned graph carries no ``skeleton_image``.
    """
    ndim = len(shape)
    if spacing is None:
        spacing = 1
    # as skan.Skeleton stores it - the dtype carries through to summarize's
    # coord columns (integer for the default unit spacing)
    spacing_arr = (
        np.asarray(spacing) if not np.isscalar(spacing) else np.full(ndim, spacing)
    )
    step_spacing = np.ones(ndim, dtype=np.float64) * spacing_arr
    coordinates = np.asarray(coordinates, dtype=np.intp).reshape(-1, ndim)
    if len(coordinates) < 2:
        return None
    linear = np.ravel_multi_index(coordinates.T, shape)
    order = np.argsort(linear)
    linear, coordinates = linear[order], coordinates[order]

    rows, cols, lengths = [], [], []
    for offset in itertools.product((-1, 0, 1), repeat=ndim):
        if not any(offset):
            continue
        neighbors = coordinates + offset
        inside = np.all((neighbors >= 0) & (neighbors < shape), axis=1)
        neighbor_linear = np.ravel_multi_index(neighbors[inside].T, shape)
        # position each neighbour would have among the skeleton pixels; it is
        # a skeleton pixel only if that position holds the same raveled index
        found = np.minimum(np.searchsorted(linear, neighbor_linear), linear.size - 1)
        hit = linear[found] == neighbor_linear
        rows.append(np.flatnonzero(inside)[hit])
        cols.append(found[hit])
        length = np.sqrt(np.sum((np.asarray(offset) * step_spacing) ** 2))
        lengths.append(np.full(int(hit.sum()), length))

    n = linear.size
    adjacency = sparse.coo_matrix(
        (np.concatenate(lengths), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    ).tocsr()
    if adjacency.nnz == 0:
        return None
    # skan's own step after building the pixel graph: replace clusters of
    # adjacent junction pixels by their minimum spanning tree
    adjacency = _mst_junctions(adjacency)

    path_graph = PathGraph.from_graph(
        adj=adjacency,
        node_coordinates=coordinates,
        node_values=np.ones(n, dtype=np.float64),
        spacing=spacing_arr,
    )
    graph = Skeleton.from_path_graph(path_graph)
    # from_path_graph infers these from the coordinates and node values;
    # set them as `build_skeleton_graph` (a uint8 image of *shape*) would
    graph.skeleton_shape = tuple(shape)
    graph.skeleton_dtype = np.dtype(np.uint8)
    return graph


def skeleton_image_from_graph(graph: Skeleton) -> np.ndarray:
    """Binary ``uint8`` skeleton image of every pixel in *graph*."""
    image = np.zeros(graph.skeleton_shape, dtype=np.uint8)
    image[tuple(np.asarray(graph.coordinates, dtype=np.intp).T)] = 1
    return image
