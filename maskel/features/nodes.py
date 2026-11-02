"""Per-node features: position, degree and type of every graph node."""

from __future__ import annotations

import numpy as np
from skan import Skeleton


def extract_node_features(
    graph: Skeleton,
    branch_data,
    *,
    radius: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Position, degree, type and (with *radius*) radius of every voxel of *graph*."""
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

        if radius is not None:
            record["radius"] = float(radius[node_id])

        records.append(record)

    return records
