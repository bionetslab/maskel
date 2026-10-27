"""Per-node features: position, degree and type of every graph node."""

from __future__ import annotations

import numpy as np
from skan import Skeleton


def extract_node_features(
    graph: Skeleton,
    branch_data,
    *,
    radius_matrix: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Extract per-node features from the skeleton graph.

    Returns one dict per graph node, with keys:
    ``node_id``, ``coord_0`` … ``coord_{ndim-1}``, ``degree``,
    ``is_endpoint``, ``is_junction``, ``is_pass_through``,
    and ``radius`` (if *radius_matrix* provided).
    """
    n_nodes = graph.coordinates.shape[0]
    ndim = graph.coordinates.shape[1]

    records: list[dict[str, object]] = []
    for node_id in range(n_nodes):
        degree = int(graph.degrees[node_id])
        coords = graph.coordinates[node_id]

        record: dict[str, object] = {
            "node_id": node_id,
            "degree": degree,
            "is_endpoint": degree == 1,
            "is_junction": degree >= 3,
            "is_pass_through": degree == 2,
        }

        for d in range(ndim):
            record[f"coord_{d}"] = int(coords[d])

        if radius_matrix is not None:
            record["radius"] = float(radius_matrix[tuple(coords)])

        records.append(record)

    return records
