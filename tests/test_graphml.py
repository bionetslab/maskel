"""Tests for maskel.graphml GraphML and pickled-networkx export."""

import pickle
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
from skan import Skeleton

from maskel.graphml import build_networkx_graph, write_graphml, write_networkx_pickle


def _junction_endpoint_count(graph: Skeleton) -> int:
    """Number of skan nodes that are junctions or endpoints (degree != 2)."""
    return int((graph.degrees != 2).sum())


class TestBuildNetworkxGraph:
    def test_nodes_are_junctions_or_endpoints(self, cross_graph):
        graph, branch_data = cross_graph
        G = build_networkx_graph(graph, branch_data)

        assert isinstance(G, nx.MultiGraph)
        assert G.number_of_nodes() == _junction_endpoint_count(graph)
        assert G.number_of_edges() == len(branch_data)
        assert all(int(graph.degrees[n]) != 2 for n in G.nodes())

    def test_node_attributes(self, cross_graph):
        graph, branch_data = cross_graph
        G = build_networkx_graph(graph, branch_data)

        node_id, data = next(iter(G.nodes(data=True)))
        assert data["coord_0"] == int(graph.coordinates[node_id, 0])
        assert data["coord_1"] == int(graph.coordinates[node_id, 1])
        assert data["degree"] == int(graph.degrees[node_id])

        endpoints = [data for _, data in G.nodes(data=True) if data["is_endpoint"]]
        assert len(endpoints) >= 2

    def test_node_radius_absent_without_radius_matrix(self, cross_graph):
        graph, branch_data = cross_graph
        G = build_networkx_graph(graph, branch_data)
        assert all("radius" not in data for _, data in G.nodes(data=True))

    def test_node_radius_sampled_when_provided(self, cross_graph, cross_skel):
        graph, branch_data = cross_graph
        radius_matrix = np.full(cross_skel.shape, 2.5, dtype=np.float64)
        G = build_networkx_graph(graph, branch_data, radius_matrix=radius_matrix)
        for node_id, data in G.nodes(data=True):
            assert data["radius"] == 2.5
            assert data["radius"] == float(
                radius_matrix[tuple(graph.coordinates[node_id])]
            )

    def test_edge_attributes_and_exclusions(self, cross_graph):
        graph, branch_data = cross_graph
        G = build_networkx_graph(graph, branch_data)

        _, _, data = next(iter(G.edges(data=True)))
        assert "branch-distance" in data
        assert "euclidean-distance" in data
        assert "node-id-src" not in data
        assert "node-id-dst" not in data
        assert not any(k.startswith("coord-") for k in data)
        assert not any(k.startswith("image-coord-") for k in data)

    def test_parallel_branches_preserved(self, loop_graph):
        graph, branch_data = loop_graph
        pairs = list(zip(branch_data["node-id-src"], branch_data["node-id-dst"]))
        assert len(pairs) > len(set(pairs))  # fixture has parallel branches

        G = build_networkx_graph(graph, branch_data)
        assert G.number_of_edges() == len(pairs)

    def test_nan_attributes_dropped(self, cross_graph):
        graph, branch_data = cross_graph
        branch_data = branch_data.copy()
        branch_data["tortuosity"] = np.nan
        branch_data["straightness"] = np.inf
        G = build_networkx_graph(graph, branch_data)

        for _, _, data in G.edges(data=True):
            assert "tortuosity" not in data
            assert "straightness" not in data

    def test_summary_features_attached_and_nan_dropped(self, cross_graph):
        graph, branch_data = cross_graph
        G = build_networkx_graph(
            graph,
            branch_data,
            summary_features={"num_nodes": 5.0, "mean_tortuosity": np.nan},
        )
        assert G.graph["num_nodes"] == 5.0
        assert "mean_tortuosity" not in G.graph

    def test_3d_skeleton(self, cross_volume_graph):
        graph, branch_data = cross_volume_graph
        G = build_networkx_graph(graph, branch_data)

        assert G.number_of_nodes() == _junction_endpoint_count(graph)
        _, data = next(iter(G.nodes(data=True)))
        assert "coord_2" in data

    def test_graphml_safe_false_keeps_nan_attributes(self, cross_graph):
        graph, branch_data = cross_graph
        branch_data = branch_data.copy()
        branch_data["tortuosity"] = np.nan
        G = build_networkx_graph(graph, branch_data, graphml_safe=False)

        for _, _, data in G.edges(data=True):
            assert "tortuosity" in data
            assert np.isnan(data["tortuosity"])

    def test_graphml_safe_false_keeps_native_int_types(self, cross_graph):
        graph, branch_data = cross_graph
        G = build_networkx_graph(
            graph, branch_data, summary_features={"object_id": 1}, graphml_safe=False
        )
        assert type(G.graph["object_id"]) is int

        _, data = next(iter(G.nodes(data=True)))
        assert type(data["degree"]) is int

    def test_graphml_safe_true_still_drops_nan_by_default(self, cross_graph):
        graph, branch_data = cross_graph
        branch_data = branch_data.copy()
        branch_data["tortuosity"] = np.nan
        G = build_networkx_graph(graph, branch_data)  # graphml_safe defaults to True

        for _, _, data in G.edges(data=True):
            assert "tortuosity" not in data


class TestWriteGraphml:
    def test_write_and_read_round_trip(self, tmp_path, cross_graph, cross_skel):
        graph, branch_data = cross_graph
        radius_matrix = np.full(cross_skel.shape, 3.0, dtype=np.float64)
        path = tmp_path / "img_graph.graphml"
        write_graphml(
            graph,
            branch_data,
            path,
            summary_features={"total_length": 42.0},
            radius_matrix=radius_matrix,
        )

        G = nx.read_graphml(str(path), node_type=int)
        assert G.number_of_nodes() == _junction_endpoint_count(graph)
        assert G.number_of_edges() == len(branch_data)
        assert G.graph["total_length"] == 42.0

        node_id, data = next(iter(G.nodes(data=True)))
        assert data["coord_0"] == int(graph.coordinates[node_id, 0])
        assert data["radius"] == 3.0
        assert "degree" in data
        assert "branch-distance" in next(iter(G.edges(data=True)))[2]

    def test_parallel_edges_round_trip_as_multigraph(self, tmp_path, loop_graph):
        graph, branch_data = loop_graph
        path = tmp_path / "loop_graph.graphml"
        write_graphml(graph, branch_data, path)

        G = nx.read_graphml(str(path), node_type=int)
        assert isinstance(G, nx.MultiGraph)
        assert G.number_of_edges() == len(branch_data)

    def test_empty_branch_data(self, tmp_path, cross_graph):
        graph, branch_data = cross_graph
        path = tmp_path / "empty_graph.graphml"
        write_graphml(graph, branch_data.iloc[0:0], path)
        G = nx.read_graphml(str(path), node_type=int)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0

    def test_yfiles_node_geometry_positions_nodes(self, tmp_path, cross_graph):
        """Without yFiles geometry, yEd stacks every node at one spot and the
        graph renders as a single square instead of the actual skeleton."""
        graph, branch_data = cross_graph
        path = tmp_path / "img_graph.graphml"
        write_graphml(graph, branch_data, path)

        tree = ET.parse(path)
        ns = {
            "g": "http://graphml.graphdrawing.org/xmlns",
            "y": "http://www.yworks.com/xml/graphml",
        }
        geometries = {
            int(node_el.get("id")): node_el.find(".//y:Geometry", ns)
            for node_el in tree.findall(".//g:node", ns)
        }
        assert len(geometries) == _junction_endpoint_count(graph)
        assert all(geom is not None for geom in geometries.values())

        positions = {
            (float(g.get("x")), float(g.get("y"))) for g in geometries.values()
        }
        assert len(positions) == len(geometries)  # nodes aren't stacked on each other
        for node_id, geom in geometries.items():
            row, col = graph.coordinates[node_id, 0], graph.coordinates[node_id, 1]
            assert float(geom.get("y")) == row
            assert float(geom.get("x")) == col

    def test_no_long_attribute_types(self, tmp_path, cross_graph, cross_skel):
        """yEd fails to import GraphML attributes typed ``long``; plain Python
        ints must be written as ``int`` instead (networkx's default for
        numpy int32/64, but ``long`` for plain Python ``int``)."""
        graph, branch_data = cross_graph
        radius_matrix = np.full(cross_skel.shape, 3.0, dtype=np.float64)
        path = tmp_path / "img_graph.graphml"
        write_graphml(graph, branch_data, path, radius_matrix=radius_matrix)

        tree = ET.parse(path)
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        attr_types = {
            key_el.get("attr.type") for key_el in tree.findall(".//g:key", ns)
        }
        assert "long" not in attr_types

    def test_no_long_attribute_type_for_int_summary_feature(
        self, tmp_path, cross_graph
    ):
        """Graph-level (summary_features) ints, e.g. ``object_id``, must also
        avoid the ``long`` attribute type yEd fails to import."""
        graph, branch_data = cross_graph
        path = tmp_path / "img_graph.graphml"
        write_graphml(graph, branch_data, path, summary_features={"object_id": 1})

        tree = ET.parse(path)
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        attr_types = {
            key_el.get("attr.type") for key_el in tree.findall(".//g:key", ns)
        }
        assert "long" not in attr_types

    def test_edge_ids_are_unique(self, tmp_path, cross_graph):
        """networkx assigns each edge its MultiGraph key (0, 1, ... *per node
        pair*) as its GraphML id, so distinct edges between different node
        pairs all collide on id="0". yEd looks up edges by id and drops all
        but one of a colliding set, hiding most of the graph."""
        graph, branch_data = cross_graph
        path = tmp_path / "img_graph.graphml"
        write_graphml(graph, branch_data, path)

        tree = ET.parse(path)
        ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
        ids = [edge_el.get("id") for edge_el in tree.findall(".//g:edge", ns)]
        assert len(ids) == len(branch_data)
        assert len(set(ids)) == len(ids)

    def test_yfiles_edge_geometry_present(self, tmp_path, cross_graph):
        """Without yFiles edgegraphics data, yEd imports the edges but never
        draws them, so the skeleton looks like disconnected node squares."""
        graph, branch_data = cross_graph
        path = tmp_path / "img_graph.graphml"
        write_graphml(graph, branch_data, path)

        tree = ET.parse(path)
        ns = {
            "g": "http://graphml.graphdrawing.org/xmlns",
            "y": "http://www.yworks.com/xml/graphml",
        }
        edge_shapes = [
            edge_el.find(".//y:PolyLineEdge", ns)
            for edge_el in tree.findall(".//g:edge", ns)
        ]
        assert len(edge_shapes) == len(branch_data)
        assert all(shape is not None for shape in edge_shapes)


class TestWriteNetworkxPickle:
    def test_write_and_read_round_trip(self, tmp_path, cross_graph, cross_skel):
        graph, branch_data = cross_graph
        radius_matrix = np.full(cross_skel.shape, 3.0, dtype=np.float64)
        path = tmp_path / "img_graph.pkl"
        write_networkx_pickle(
            graph,
            branch_data,
            path,
            summary_features={"total_length": 42.0},
            radius_matrix=radius_matrix,
        )

        with open(path, "rb") as f:
            G = pickle.load(f)

        assert isinstance(G, nx.MultiGraph)
        assert G.number_of_nodes() == _junction_endpoint_count(graph)
        assert G.number_of_edges() == len(branch_data)
        assert G.graph["total_length"] == 42.0

        node_id, data = next(iter(G.nodes(data=True)))
        assert data["coord_0"] == int(graph.coordinates[node_id, 0])
        assert data["radius"] == 3.0
        assert type(data["degree"]) is int

    def test_nan_attributes_survive_the_round_trip(self, tmp_path, cross_graph):
        graph, branch_data = cross_graph
        branch_data = branch_data.copy()
        branch_data["tortuosity"] = np.nan
        path = tmp_path / "img_graph.pkl"
        write_networkx_pickle(graph, branch_data, path)

        with open(path, "rb") as f:
            G = pickle.load(f)

        for _, _, data in G.edges(data=True):
            assert "tortuosity" in data
            assert np.isnan(data["tortuosity"])

    def test_parallel_edges_round_trip_as_multigraph(self, tmp_path, loop_graph):
        graph, branch_data = loop_graph
        path = tmp_path / "loop_graph.pkl"
        write_networkx_pickle(graph, branch_data, path)

        with open(path, "rb") as f:
            G = pickle.load(f)
        assert isinstance(G, nx.MultiGraph)
        assert G.number_of_edges() == len(branch_data)

    def test_empty_branch_data(self, tmp_path, cross_graph):
        graph, branch_data = cross_graph
        path = tmp_path / "empty_graph.pkl"
        write_networkx_pickle(graph, branch_data.iloc[0:0], path)

        with open(path, "rb") as f:
            G = pickle.load(f)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
