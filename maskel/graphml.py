"""GraphML export for vessel skeleton graphs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
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

# Plain GraphML carries no layout information, so yEd places every node at
# the same default position and they render as one overlapping square. This
# namespace/key pair adds the yFiles visualization block yEd actually reads
# for node placement and size.
_GRAPHML_NS = "http://graphml.graphdrawing.org/xmlns"
_YFILES_NS = "http://www.yworks.com/xml/graphml"
_YFILES_NODE_KEY = "yfiles_node_gfx"
_YFILES_EDGE_KEY = "yfiles_edge_gfx"
_DEFAULT_NODE_SIZE = 10.0


def _is_missing(value: object) -> bool:
    """True for NaN/Inf floats, which GraphML cannot represent."""
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value) or np.isinf(value))
    return False


def _as_graphml_value(value: object) -> object:
    """Coerce plain Python ints to a fixed-width type.

    networkx's GraphML writer maps plain Python ``int`` to the GraphML
    ``long`` attribute type, which yEd fails to import. numpy integer types
    (e.g. ``int32``) map to ``int`` instead, which yEd accepts.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return np.int32(value)
    return value


def _edge_attributes(row: dict[str, object]) -> dict[str, object]:
    return {
        key: _as_graphml_value(value)
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
            G.graph[summary[0]] = _as_graphml_value(summary[1])

    for node_id in G.nodes():
        coords = graph.coordinates[node_id]
        degree = int(graph.degrees[node_id])
        attrs = {f"coord_{d}": _as_graphml_value(int(c)) for d, c in enumerate(coords)}
        attrs["degree"] = _as_graphml_value(degree)
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


def _add_yfiles_geometry(G: nx.MultiGraph, path: Path) -> None:
    """Inject yFiles visualization data so yEd lays out the graph on open.

    Reads back the file ``nx.write_graphml`` just wrote and adds:

    - a ``y:ShapeNode``/``y:Geometry`` block per node, positioned from its
      ``coord_*`` attributes (last two axes = row, col) and sized from
      ``radius`` when present. Without this, yEd has no placement info and
      stacks every node at the same spot, rendering as a single yellow
      square.
    - a ``y:PolyLineEdge`` block per edge. yEd only draws an edge as a line
      between its nodes if it carries this visualization data; without it,
      edges are imported (the topology is intact) but simply not rendered.
    - a unique ``id`` per edge element, replacing networkx's MultiGraph-key
      id (0, 1, ... *per node pair*, so most edges in a graph collide on
      id="0"). yEd looks up edges by id and drops all but one of a colliding
      set, so without this only one edge in the whole graph is ever shown.
    """
    ET.register_namespace("", _GRAPHML_NS)
    ET.register_namespace("y", _YFILES_NS)
    tree = ET.parse(path)
    root = tree.getroot()

    node_key_el = ET.Element(
        f"{{{_GRAPHML_NS}}}key",
        {"for": "node", "id": _YFILES_NODE_KEY, "yfiles.type": "nodegraphics"},
    )
    edge_key_el = ET.Element(
        f"{{{_GRAPHML_NS}}}key",
        {"for": "edge", "id": _YFILES_EDGE_KEY, "yfiles.type": "edgegraphics"},
    )
    root.insert(0, edge_key_el)
    root.insert(0, node_key_el)

    graph_el = root.find(f"{{{_GRAPHML_NS}}}graph")
    for node_el in graph_el.findall(f"{{{_GRAPHML_NS}}}node"):
        node_id = int(node_el.get("id"))
        attrs = G.nodes[node_id]
        n_coords = sum(1 for k in attrs if k.startswith("coord_"))
        y = attrs[f"coord_{n_coords - 2}"]
        x = attrs[f"coord_{n_coords - 1}"]
        size = (
            max(2.0 * attrs["radius"], 4.0) if "radius" in attrs else _DEFAULT_NODE_SIZE
        )

        data_el = ET.SubElement(
            node_el, f"{{{_GRAPHML_NS}}}data", {"key": _YFILES_NODE_KEY}
        )
        shape_el = ET.SubElement(data_el, f"{{{_YFILES_NS}}}ShapeNode")
        ET.SubElement(
            shape_el,
            f"{{{_YFILES_NS}}}Geometry",
            {
                "height": f"{size:.2f}",
                "width": f"{size:.2f}",
                "x": f"{x:.2f}",
                "y": f"{y:.2f}",
            },
        )
        ET.SubElement(
            shape_el,
            f"{{{_YFILES_NS}}}Fill",
            {"color": "#FFCC00", "transparent": "false"},
        )
        ET.SubElement(
            shape_el,
            f"{{{_YFILES_NS}}}BorderStyle",
            {"color": "#000000", "type": "line", "width": "1.0"},
        )

    for i, edge_el in enumerate(graph_el.findall(f"{{{_GRAPHML_NS}}}edge")):
        # networkx sets an edge's id to its MultiGraph key (0, 1, ...), which
        # only disambiguates parallel edges between the *same* node pair, so
        # most edges in the file end up sharing id="0". yEd looks up edges by
        # id and drops all but one of a colliding set, hiding most edges.
        edge_el.set("id", f"e{i}")
        data_el = ET.SubElement(
            edge_el, f"{{{_GRAPHML_NS}}}data", {"key": _YFILES_EDGE_KEY}
        )
        edge_shape_el = ET.SubElement(data_el, f"{{{_YFILES_NS}}}PolyLineEdge")
        ET.SubElement(
            edge_shape_el,
            f"{{{_YFILES_NS}}}LineStyle",
            {"color": "#000000", "type": "line", "width": "1.0"},
        )
        ET.SubElement(
            edge_shape_el,
            f"{{{_YFILES_NS}}}Arrows",
            {"source": "none", "target": "none"},
        )

    tree.write(path, xml_declaration=True, encoding="UTF-8")


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
    if G.number_of_nodes() > 0:
        _add_yfiles_geometry(G, Path(path))
