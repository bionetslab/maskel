"""Skeleton image -> skan graph, the basis of every branch and node feature."""

from __future__ import annotations

import numpy as np
from skan import Skeleton

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
