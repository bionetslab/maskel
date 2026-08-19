"""GraphML export for vessel skeleton graphs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from skan.csr import skeleton_to_nx

if TYPE_CHECKING:
    from skan import Skeleton

# Branch-table columns describing topology or node positions; excluded from
# edge attributes (node positions live on the node elements instead).
_EXCLUDED_COLUMNS = {
    "node-id-src",
    "node-id-dst",
    "skeleton-id",
}


def _is_missing(value: object) -> bool:
    """True for NaN/Inf floats, which GraphML cannot represent."""
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value) or np.isinf(value))
    return False


def _edge_attributes(row: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("coord-")
        and not key.startswith("image-coord-")
        and key not in _EXCLUDED_COLUMNS
        and not _is_missing(value)
    }


def build_networkx_graph(
    graph: Skeleton,
    branch_data,
    summary_features: dict[str, float] | None = None,
    radius_matrix: np.ndarray | None = None,
) -> nx.MultiGraph:
    """Convert a skan skeleton graph into a networkx MultiGraph.

    Nodes carry coordinates, degree, node-type flags, and ``radius`` (when
    *radius_matrix* is provided). Edges carry the per-branch features from
    *branch_data*. NaN/Inf attributes are omitted. *summary_features* is
    attached as graph-level attributes.

    Parameters
    ----------
    graph : Skeleton
        Pre-built skan Skeleton graph (e.g. from ``build_vessel_graph``).
    branch_data : DataFrame
        Pre-computed branch summary (e.g. from ``skan.summarize``).
    summary_features : dict[str, float], optional
        Per-image summary features attached as graph-level attributes.
    radius_matrix : ndarray, optional
        EDT radius array from ``compute_radii``; radius at each node is
        attached as a ``radius`` node attribute when provided.
    """
    G = skeleton_to_nx(graph, branch_data)
    G.graph.clear()

    for summary in (summary_features or {}).items():
        if not _is_missing(summary[1]):
            G.graph[summary[0]] = summary[1]

    for node_id in G.nodes():
        coords = graph.coordinates[node_id]
        degree = int(graph.degrees[node_id])
        attrs = {f"coord_{d}": int(c) for d, c in enumerate(coords)}
        attrs["degree"] = degree
        attrs["is_endpoint"] = degree == 1
        attrs["is_junction"] = degree >= 3
        attrs["is_pass_through"] = degree == 2
        if radius_matrix is not None:
            attrs["radius"] = float(radius_matrix[tuple(coords)])
        G.nodes[node_id].update(attrs)

    G.clear_edges()
    for row in branch_data.to_dict(orient="records"):
        G.add_edge(
            int(row["node-id-src"]),
            int(row["node-id-dst"]),
            **_edge_attributes(row),
        )

    return G


def write_graphml(
    graph: Skeleton,
    branch_data,
    path: str | Path,
    *,
    summary_features: dict[str, float] | None = None,
    radius_matrix: np.ndarray | None = None,
) -> None:
    """Write a skeleton graph to a GraphML file.

    Parameters
    ----------
    graph : Skeleton
        Pre-built skan Skeleton graph (e.g. from ``build_vessel_graph``).
    branch_data : DataFrame
        Pre-computed branch summary (e.g. from ``skan.summarize``).
    path : str or Path
        Destination file path.
    summary_features : dict[str, float], optional
        Per-image summary features attached as graph-level attributes.
    radius_matrix : ndarray, optional
        EDT radius array from ``compute_radii``; radius at each node is
        attached as a ``radius`` node attribute when provided.
    """
    G = build_networkx_graph(
        graph,
        branch_data,
        summary_features=summary_features,
        radius_matrix=radius_matrix,
    )
    nx.write_graphml(G, str(path))
