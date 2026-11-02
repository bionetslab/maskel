import networkx as nx
import numpy as np
import pytest
from skan import summarize
from skan.csr import skeleton_to_nx

from maskel.graph import build_graph
from maskel.postprocessing import collapse_triangle_junctions, prune_short_spurs


def _graph(img, spacing=None):
    """Graph of a skeleton image, built from its coordinates."""
    return build_graph(np.argwhere(img), img.shape, spacing)


def _image(coords, shape):
    img = np.zeros(shape, dtype=np.uint8)
    img[tuple(np.asarray(coords).T)] = 1
    return img


def _n_cycles(graph) -> int:
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
        graph = _graph(img)
        assert _n_cycles(graph) == 1

        # perimeter 16 < 2.5 * (2 * 5): small relative to the local diameter
        coords, _ = collapse_triangle_junctions(graph, np.full(int(img.sum()), 5.0))

        cleaned = build_graph(coords, img.shape)
        assert _n_cycles(cleaned) == 0
        branch_data = summarize(cleaned, separator="-")
        assert len(branch_data) == 4
        assert (branch_data["branch-type"] == 1).all()  # four tip-junction arms

    def test_asymmetric_cycle_collapses_to_its_centroid_keeping_every_arm(self):
        # 4x4 hollow square with arms leaving three corners. networkx reports
        # some of its branches against their path direction, which once put
        # cycle nodes at the far end of a branch and dropped the left arm.
        img = np.zeros((40, 40), dtype=np.uint8)
        img[18:22, 18] = img[18:22, 21] = img[18, 18:22] = img[21, 18:22] = 1
        img[18, 4:18] = 1
        img[21, 22:36] = 1
        img[4:18, 21] = 1
        graph = _graph(img)

        coords, _ = collapse_triangle_junctions(graph, np.full(int(img.sum()), 5.0))

        cleaned = build_graph(coords, img.shape)
        assert cleaned.coordinates[cleaned.degrees >= 3].tolist() == [[19, 20]]
        assert sorted(cleaned.coordinates[cleaned.degrees == 1].tolist()) == [
            [4, 21],
            [18, 4],
            [21, 35],
        ]

    def test_kept_voxels_keep_radii_and_new_ones_are_estimated(self):
        img = _ring_junction_image()
        graph = _graph(img)
        radius = np.full(len(graph.coordinates), 5.0)
        radius[np.all(graph.coordinates == (20, 5), axis=1)] = 7.0

        coords, new_radius = collapse_triangle_junctions(graph, radius)

        old = {tuple(c): r for c, r in zip(graph.coordinates.tolist(), radius)}
        for c, r in zip(coords.tolist(), new_radius):
            assert r == old.get(tuple(c), 5.0)  # centroid: mean of the cycle radii

    def test_large_cycle_is_kept(self):
        img = _ring_junction_image()
        graph = _graph(img)
        coords, _ = collapse_triangle_junctions(graph, np.zeros(int(img.sum())))
        np.testing.assert_array_equal(coords, graph.coordinates)

    def test_fully_collapsible_ring_leaves_nothing(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        img[8:12, 8] = 1
        img[8:12, 11] = 1
        img[8, 8:12] = 1
        img[11, 8:12] = 1

        coords, radius = collapse_triangle_junctions(
            _graph(img), np.ones(int(img.sum())), threshold_factor=100.0
        )
        assert len(coords) == 0 and len(radius) == 0


class TestPruneShortSpurs:
    def _cross_with_spur(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 4:29] = 1  # long horizontal line through the junction
        img[13:17, 16] = 1  # short vertical spur (length 3) off the junction
        return img

    def test_short_spur_is_removed_but_junction_voxel_kept(self):
        img = self._cross_with_spur()
        graph = _graph(img)
        cleaned = _image(graph.coordinates[prune_short_spurs(graph, 10.0)], img.shape)

        assert cleaned[16, 16] == 1
        np.testing.assert_array_equal(cleaned[16, 4:29], img[16, 4:29])
        assert cleaned[13:16, 16].sum() == 0

    @pytest.mark.parametrize("min_length", [2.0, 10.0])
    def test_long_branches_and_short_thresholds_keep_everything(self, min_length):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, :] = 1
        img[:, 16] = 1
        assert prune_short_spurs(_graph(img), min_length).all()
        assert prune_short_spurs(_graph(self._cross_with_spur()), 2.0).all()

    def test_all_short_spurs_removed_leaves_nothing(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 8:24] = 1
        img[8:24, 16] = 1
        assert not prune_short_spurs(_graph(img), 10.0).any()

    def test_junction_junction_branch_is_not_a_spur(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        img[4, 4:10] = 1
        img[4:10, 4] = 1
        img[9, 4:10] = 1
        img[4:10, 9] = 1
        assert prune_short_spurs(_graph(img), 10.0).all()
