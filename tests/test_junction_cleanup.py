import networkx as nx
import numpy as np
import pandas as pd
from skan import Skeleton, summarize
from skan.csr import skeleton_to_nx

from maskel.features import build_skeleton_graph, skeleton_image_from_graph
from maskel.postprocessing import collapse_triangle_junctions

from ._helpers import cross_image


def _n_cycles(graph: Skeleton) -> int:
    junction_graph = skeleton_to_nx(graph, summarize(graph, separator="-"))
    return len(nx.cycle_basis(nx.Graph(junction_graph)))


def _ring_junction_image(c: int = 20, size: int = 40) -> np.ndarray:
    """Cross whose centre is a 5x5 hollow square: one small cycle with four
    arms attached at the middle of its sides."""
    img = np.zeros((size, size), dtype=np.uint8)
    img[c - 2 : c + 3, c - 2] = 1
    img[c - 2 : c + 3, c + 2] = 1
    img[c - 2, c - 2 : c + 3] = 1
    img[c + 2, c - 2 : c + 3] = 1
    img[c, 4 : c - 2] = 1
    img[c, c + 3 : size - 4] = 1
    img[4 : c - 2, c] = 1
    img[c + 3 : size - 4, c] = 1
    return img


class TestCollapseTriangleJunctions:
    def test_small_cycle_collapses_to_single_junction(self):
        img = _ring_junction_image()
        graph = build_skeleton_graph(img)
        assert _n_cycles(graph) == 1

        # perimeter 16 < 2.5 * (2 * 5): small relative to the local diameter
        cleaned = collapse_triangle_junctions(
            graph, radius_matrix=np.where(img > 0, 5.0, 0.0)
        )

        assert cleaned is not graph
        assert _n_cycles(cleaned) == 0
        branch_data = summarize(cleaned, separator="-")
        assert len(branch_data) == 4
        assert (branch_data["branch-type"] == 1).all()  # four tip-junction arms

    def test_rebuilt_graph_matches_graph_of_redrawn_image(self):
        img = _ring_junction_image()
        graph = build_skeleton_graph(img, spacing=(2.0, 0.5))
        cleaned = collapse_triangle_junctions(
            graph, radius_matrix=np.where(img > 0, 5.0, 0.0)
        )

        redrawn = build_skeleton_graph(
            skeleton_image_from_graph(cleaned), spacing=(2.0, 0.5)
        )
        pd.testing.assert_frame_equal(
            summarize(cleaned, separator="-"),
            summarize(redrawn, separator="-"),
            check_exact=True,
        )

    def test_large_cycle_is_kept_and_graph_returned_unchanged(self):
        img = _ring_junction_image()
        graph = build_skeleton_graph(img)

        # without radii the local diameter defaults to 1 pixel
        assert collapse_triangle_junctions(graph) is graph

    def test_acyclic_graph_returned_unchanged(self):
        graph = build_skeleton_graph(cross_image())
        assert collapse_triangle_junctions(graph, threshold_factor=100.0) is graph

    def test_fully_collapsible_ring_returns_none(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[8:12, 8] = 1
        img[8:12, 11] = 1
        img[8, 8:12] = 1
        img[11, 8:12] = 1

        graph = build_skeleton_graph(img)
        assert collapse_triangle_junctions(graph, threshold_factor=100.0) is None
