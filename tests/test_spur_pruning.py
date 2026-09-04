import numpy as np
from skan import summarize

from maskel.features import build_skeleton_graph
from maskel.spur_pruning import prune_short_spurs


class TestPruneShortSpurs:
    def test_short_spur_is_removed_but_junction_pixel_kept(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 4:29] = 1  # long horizontal line through the junction
        img[13:17, 16] = 1  # short vertical spur (length 3) off the junction

        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-")

        cleaned = prune_short_spurs(img, graph, branch_data, min_length=10.0)

        assert cleaned[16, 16] == 1, (
            "junction pixel must stay so the rest stays connected"
        )
        assert np.array_equal(cleaned[16, 4:29], img[16, 4:29]), (
            "main line must survive"
        )
        assert cleaned[13:16, 16].sum() == 0, "the whole spur must be removed"

    def test_long_branches_are_not_touched(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, :] = 1
        img[:, 16] = 1

        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-")

        cleaned = prune_short_spurs(img, graph, branch_data, min_length=10.0)

        assert np.array_equal(cleaned, img)

    def test_all_short_spurs_removed_yields_empty_skeleton(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 8:24] = 1
        img[8:24, 16] = 1

        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-")

        cleaned = prune_short_spurs(img, graph, branch_data, min_length=10.0)

        assert not cleaned.any()
        assert cleaned.dtype == np.uint8

    def test_junction_junction_branch_is_not_a_spur(self):
        # a triangle: three junction nodes, no endpoints at all.
        img = np.zeros((16, 16), dtype=np.uint8)
        img[4, 4:10] = 1
        img[4:10, 4] = 1
        img[9, 4:10] = 1
        img[4:10, 9] = 1

        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-")

        cleaned = prune_short_spurs(img, graph, branch_data, min_length=10.0)

        assert np.array_equal(cleaned, img)

    def test_empty_branch_data_returns_copy(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        img[3, 3:6] = 1

        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-").iloc[0:0]
        assert branch_data.empty

        cleaned = prune_short_spurs(img, graph, branch_data, min_length=10.0)

        assert np.array_equal(cleaned, img)
        assert cleaned is not img

    def test_min_length_threshold_is_respected(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 4:29] = 1
        img[13:17, 16] = 1  # spur of length 3

        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-")

        # threshold below the spur's actual length -> nothing qualifies
        cleaned = prune_short_spurs(img, graph, branch_data, min_length=2.0)

        assert np.array_equal(cleaned, img)
