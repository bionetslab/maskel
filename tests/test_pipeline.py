import numpy as np
import pytest
from skan import summarize

from maskel.config import ExtractionConfig, OutputConfig, PipelineConfig
from maskel.features import build_skeleton_graph
from maskel.pipeline import AnalysisResult, ObjectResult, analyze_segmentation_mask

from ._helpers import cross_volume, line_volume


class TestAnalysisResult:
    def test_default_radius_matrix_is_none(self):
        result = AnalysisResult(
            skeleton=np.zeros((2, 2), dtype=np.uint8),
            objects=[],
        )
        assert result.radius_matrix is None
        assert result.summary_features == []
        assert result.branch_records == []
        assert result.node_records == []

    def test_flattened_properties_combine_all_objects(self):
        skeleton = np.eye(4, dtype=np.uint8)
        radius = np.ones_like(skeleton, dtype=np.float64)
        objects = [
            ObjectResult(
                object_id=1,
                offset=(0, 0),
                summary_features={"num_nodes": 3.0, "object_id": 1},
                branch_records=[{"branch_id": 0, "object_id": 1}],
                node_records=[{"coord_0": 0, "coord_1": 0, "object_id": 1}],
            ),
            ObjectResult(
                object_id=2,
                offset=(0, 0),
                summary_features={},  # summary disabled for this object
                branch_records=[{"branch_id": 0, "object_id": 2}],
                node_records=[],
            ),
        ]
        result = AnalysisResult(
            skeleton=skeleton, objects=objects, radius_matrix=radius
        )
        assert np.array_equal(result.skeleton, skeleton)
        # only object 1 contributed a summary row - object 2's is empty
        assert result.summary_features == [{"num_nodes": 3.0, "object_id": 1}]
        assert result.branch_records == [
            {"branch_id": 0, "object_id": 1},
            {"branch_id": 0, "object_id": 2},
        ]
        assert result.node_records == [{"coord_0": 0, "coord_1": 0, "object_id": 1}]
        assert np.array_equal(result.radius_matrix, radius)


class TestAnalyzeSegmentationMask:
    @pytest.fixture
    def analysis_config(self):
        """Shared config with branches and summary enabled."""
        return PipelineConfig(
            extraction=ExtractionConfig(branches=True, summary=True),
            output=OutputConfig(),
        )

    def test_empty_image_returns_empty_result(self, analysis_config):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = analyze_segmentation_mask(img, analysis_config)
        assert not result.skeleton.any()
        assert result.summary_features == []
        assert result.branch_records == []
        assert result.radius_matrix is None
        assert result.objects == []

    def test_result_carries_object_graph(self, cross_skel, analysis_config):
        result = analyze_segmentation_mask(cross_skel, analysis_config)
        assert len(result.objects) == 1
        obj = result.objects[0]
        assert obj.object_id == 1
        assert not obj.branch_data.empty

    def test_cross_produces_skeleton(self, cross_skel, analysis_config):
        result = analyze_segmentation_mask(cross_skel, analysis_config)
        assert result.skeleton.any()
        assert result.skeleton.dtype == np.uint8
        assert result.skeleton.shape == cross_skel.shape

    def test_non_binary_input_is_binarized(self, analysis_config):
        img = np.zeros((20, 20), dtype=np.int32)
        img[10, 5:15] = 200
        result = analyze_segmentation_mask(img, analysis_config)
        assert result.skeleton.any()
        assert set(np.unique(result.skeleton)) <= {0, 1}

    def test_summary_disabled_returns_empty_features(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(summary=False),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(cross_skel, config)
        assert result.summary_features == []

    def test_binary_input_gets_object_id_one(self, cross_skel, analysis_config):
        result = analyze_segmentation_mask(cross_skel, analysis_config)
        assert result.summary_features[0]["object_id"] == 1
        assert all(r["object_id"] == 1 for r in result.branch_records)

    def test_branches_enabled(self, cross_skel, analysis_config):
        result = analyze_segmentation_mask(cross_skel, analysis_config)
        assert len(result.branch_records) > 0
        assert isinstance(result.branch_records[0], dict)

    def test_branches_disabled_returns_empty_records(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(branches=False),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(cross_skel, config)
        assert result.branch_records == []

    def test_mask_radius_enabled(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(mask_radius=True, summary=True),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(cross_skel, config)
        assert result.radius_matrix is not None
        assert result.radius_matrix.shape == cross_skel.shape
        assert result.radius_matrix.any()
        assert result.summary_features[0]["mean_radius"] > 0

    def test_mask_radius_disabled_radius_none(self, cross_skel, analysis_config):
        result = analyze_segmentation_mask(cross_skel, analysis_config)
        assert result.radius_matrix is None

    def test_radius_stats_in_summary_when_enabled(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(mask_radius=True, summary=True),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(cross_skel, config)
        radius_keys = [
            "mean_radius",
            "std_radius",
            "min_radius",
            "max_radius",
            "mean_diameter",
            "std_diameter",
            "min_diameter",
            "max_diameter",
        ]
        summary = result.summary_features[0]
        for key in radius_keys:
            assert key in summary
            assert summary[key] > 0

    def test_fractal_dimension_disabled_by_default(self, cross_skel, analysis_config):
        result = analyze_segmentation_mask(cross_skel, analysis_config)
        assert result.summary_features[0]["fractal_dimension"] == 0.0

    def test_fractal_dimension_enabled(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(fractal_dimension=True, summary=True),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(cross_skel, config)
        assert result.summary_features[0]["fractal_dimension"] > 0

    def test_all_options_enabled(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(
                branches=True,
                branch_text=True,
                summary=True,
                fractal_dimension=True,
                mask_radius=True,
            ),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(cross_skel, config)
        assert result.skeleton.any()
        assert len(result.summary_features) > 0
        assert len(result.branch_records) > 0
        assert result.radius_matrix is not None
        assert result.summary_features[0]["fractal_dimension"] > 0
        assert result.summary_features[0]["mean_radius"] > 0

    def test_all_options_disabled(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(
                branches=False,
                branch_text=False,
                summary=False,
                fractal_dimension=False,
                mask_radius=False,
            ),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(cross_skel, config)
        assert result.skeleton.any()
        assert result.radius_matrix is None
        assert result.summary_features == []
        assert result.branch_records == []

    def test_3d_image(self):
        vol = cross_volume()
        config = PipelineConfig(
            extraction=ExtractionConfig(summary=True), output=OutputConfig()
        )
        result = analyze_segmentation_mask(vol, config)
        assert result.skeleton.any()
        assert len(result.summary_features) > 0

    def test_3d_image_with_radius(self):
        vol = line_volume((12, 12, 12), axis=2)
        config = PipelineConfig(
            extraction=ExtractionConfig(
                mask_radius=True, fractal_dimension=True, summary=True
            ),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(vol, config)
        assert result.radius_matrix is not None
        assert result.radius_matrix.shape == vol.shape
        assert result.summary_features[0]["mean_radius"] > 0

    def test_cross_topology_num_endpoints(self, cross_skel, analysis_config):
        result = analyze_segmentation_mask(cross_skel, analysis_config)
        assert result.summary_features[0]["num_endpoints"] == 4

    def test_cross_topology_num_bifurcations(self, cross_skel, analysis_config):
        result = analyze_segmentation_mask(cross_skel, analysis_config)
        assert result.summary_features[0]["num_bifurcations"] == 1


def _place_cross(
    mask: np.ndarray, top_left: tuple[int, int], size: int, object_id: int
):
    r0, c0 = top_left
    mid = size // 2
    mask[r0 + mid, c0 : c0 + size] = object_id
    mask[r0 : r0 + size, c0 + mid] = object_id


class TestJunctionCleanupGuard:
    """junction_cleanup can legitimately collapse an object's entire skeleton
    (a small isolated cycle with a generous cleanup_threshold_factor) - this
    must degrade to an empty result like every other empty-skeleton path,
    not crash inside skan/scipy trying to build a graph with zero edges."""

    def test_fully_collapsible_ring_returns_empty_instead_of_crashing(self):
        # A small, isolated 4x4 ring: skan/collapse_triangle_junctions sees
        # this as a single small cycle, which a generous threshold_factor
        # collapses entirely, leaving zero edges.
        img = np.zeros((20, 20), dtype=np.uint8)
        img[8:12, 8] = 1
        img[8:12, 11] = 1
        img[8, 8:12] = 1
        img[11, 8:12] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(
                summary=True,
                junction_cleanup=True,
                cleanup_threshold_factor=100.0,
            ),
            output=OutputConfig(),
        )

        result = analyze_segmentation_mask(img, config)

        assert not result.skeleton.any()
        assert result.branch_records == []
        assert result.objects[0].summary_features == {"object_id": 1}


class TestPruneSpursIntegration:
    """config.extraction.prune_spurs pipeline integration - the standalone
    prune_short_spurs function is well covered in isolation
    (tests/test_spur_pruning.py), but nothing previously exercised the
    pipeline's own loop: repeated pruning across spur_iterations, the
    graph rebuild in between, and the prune-to-nothing early return."""

    @staticmethod
    def _no_prunable_spurs_remain(skeleton, min_length):
        """True once re-summarizing *skeleton*'s graph finds no more
        endpoint-to-junction branches shorter than *min_length* - the
        actual convergence condition prune_spurs is meant to reach,
        independent of exactly which pixels a given thinning pass leaves
        behind (Lee94 can simplify a junction pixel's own position, e.g.
        rounding a corner, without changing this topological property)."""
        branch_data = summarize(build_skeleton_graph(skeleton), separator="-")
        still_prunable = (branch_data["branch-type"] == 1) & (
            branch_data["branch-distance"] < min_length
        )
        return not still_prunable.any()

    def test_single_short_spur_removed_end_to_end(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 4:29] = 1  # long horizontal line through the junction
        img[13:17, 16] = 1  # short vertical spur (length 3) off the junction

        config = PipelineConfig(
            extraction=ExtractionConfig(
                summary=True, prune_spurs=True, min_spur_length=10.0
            ),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(img, config)

        # spur shaft, well clear of the junction, is gone; the main line's
        # far ends (equally clear of it) survive
        assert not result.skeleton[13:15, 16].any()
        assert result.skeleton[16, 4] == 1
        assert result.skeleton[16, 28] == 1
        assert self._no_prunable_spurs_remain(result.skeleton, min_length=10.0)

    def test_spur_iterations_exposes_a_second_spur(self):
        # A junction J with two short spurs (S1, S2) plus a short stub M
        # connecting J to the rest of the structure via junction J2. On the
        # first pass only S1/S2 qualify (endpoint-to-junction); removing
        # both drops J to degree 1, exposing M as a *new* endpoint-to-
        # junction spur only a second pass will remove.
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 4:30] = 1  # main line (through J2 at x=16)
        img[16:21, 16] = 1  # M: J2-to-J stub, length 4
        img[20, 16:20] = 1  # S1: short spur right of J, length 3
        img[20, 13:17] = 1  # S2: short spur left of J, length 3

        config_one_pass = PipelineConfig(
            extraction=ExtractionConfig(
                summary=True,
                prune_spurs=True,
                min_spur_length=10.0,
                spur_iterations=1,
            ),
            output=OutputConfig(),
        )
        after_one = analyze_segmentation_mask(img, config_one_pass)

        config_two_pass = PipelineConfig(
            extraction=ExtractionConfig(
                summary=True,
                prune_spurs=True,
                min_spur_length=10.0,
                spur_iterations=2,
            ),
            output=OutputConfig(),
        )
        after_two = analyze_segmentation_mask(img, config_two_pass)

        # One pass isn't enough - removing S1/S2 exposes M as a new spur a
        # single pass never reaches, so the second pass must remove more.
        assert after_two.skeleton.sum() < after_one.skeleton.sum()
        assert self._no_prunable_spurs_remain(after_two.skeleton, min_length=10.0)
        # the main line's far ends, well clear of J/J2, survive untouched
        assert after_two.skeleton[16, 4] == 1
        assert after_two.skeleton[16, 29] == 1

    def test_pruning_everything_away_returns_empty_result(self):
        # A plus shape: all four arms qualify as spurs at once, leaving
        # nothing behind - exercises the pipeline's own
        # "pruned itself away to nothing" early return.
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 8:24] = 1
        img[8:24, 16] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(
                summary=True, prune_spurs=True, min_spur_length=10.0
            ),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(img, config)

        assert not result.skeleton.any()
        assert result.branch_records == []


class TestDegenerateObjects:
    """Border-touching bounding boxes and single-pixel objects - edge cases
    in _iter_object_crops' bbox padding/clamping and in the graph-building
    guard for skeletons with no edges at all."""

    def test_object_touching_every_array_edge_is_processed_without_crashing(self):
        # Object 1 fills its own bounding box edge-to-edge against the
        # array's top-left corner, forcing _iter_object_crops' padding
        # clamp (max(start-1, 0) / min(stop+1, dim)) to actually engage on
        # multiple axes at once, rather than always having a free
        # background pixel to pad into.
        mask = np.zeros((12, 12), dtype=np.int32)
        mask[0:3, 0:3] = 1
        mask[8:10, 8:10] = 2

        config = PipelineConfig(
            extraction=ExtractionConfig(summary=True), output=OutputConfig()
        )
        result = analyze_segmentation_mask(mask, config)

        object_ids = {obj.object_id for obj in result.objects}
        assert object_ids == {1, 2}
        assert {f["object_id"] for f in result.summary_features} == {1, 2}

    def test_single_pixel_object_degrades_instead_of_crashing(self):
        # A lone foreground pixel survives Lee94 thinning as itself (it's
        # protected as an endpoint), but has no edges for skan to build a
        # graph from - must degrade to an empty-features result rather than
        # crashing inside skan/scipy building a zero-edge graph.
        img = np.zeros((16, 16), dtype=np.uint8)
        img[8, 8] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(branches=True, nodes=True, summary=True),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(img, config)

        assert result.skeleton[8, 8] == 1, "the single pixel is itself the skeleton"
        assert result.objects[0].summary_features == {"object_id": 1}
        assert result.branch_records == []
        assert result.node_records == []

    def test_mixed_component_with_one_isolated_pixel_still_builds_a_graph(self):
        # An isolated pixel alongside a real branch elsewhere in the same
        # object's crop must not take down the whole graph - only an
        # object whose *entire* skeleton is isolated points needs the
        # guard above.
        img = np.zeros((16, 16), dtype=np.uint8)
        img[2, 2:5] = 1  # a real 3px branch
        img[12, 12] = 1  # an unrelated isolated pixel, same object

        config = PipelineConfig(
            extraction=ExtractionConfig(branches=True, summary=True),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(img, config)

        assert result.skeleton[12, 12] == 1
        assert len(result.branch_records) > 0


class TestMultiObject:
    @pytest.fixture
    def two_objects_mask(self) -> np.ndarray:
        """40x40 canvas with two disjoint 12px crosses, labeled 5 and 9."""
        mask = np.zeros((40, 40), dtype=np.uint8)
        _place_cross(mask, (2, 2), 12, object_id=5)
        _place_cross(mask, (24, 24), 12, object_id=9)
        return mask

    @pytest.fixture
    def full_config(self):
        return PipelineConfig(
            extraction=ExtractionConfig(branches=True, nodes=True, summary=True),
            output=OutputConfig(),
        )

    def test_one_summary_row_per_object(self, two_objects_mask, full_config):
        result = analyze_segmentation_mask(two_objects_mask, full_config)
        assert len(result.summary_features) == 2
        assert {row["object_id"] for row in result.summary_features} == {5, 9}

    def test_object_id_matches_mask_label_not_renumbered(
        self, two_objects_mask, full_config
    ):
        result = analyze_segmentation_mask(two_objects_mask, full_config)
        assert {r["object_id"] for r in result.branch_records} == {5, 9}
        assert {r["object_id"] for r in result.node_records} == {5, 9}

    def test_node_coords_are_in_global_image_frame(self, two_objects_mask, full_config):
        result = analyze_segmentation_mask(two_objects_mask, full_config)
        object_9_rows = [r for r in result.node_records if r["object_id"] == 9]
        assert object_9_rows
        for r in object_9_rows:
            assert r["coord_0"] >= 23
            assert r["coord_1"] >= 23

    def test_objects_tagged_and_offset(self, two_objects_mask, full_config):
        result = analyze_segmentation_mask(two_objects_mask, full_config)
        by_id = {obj.object_id: obj for obj in result.objects}
        assert set(by_id) == {5, 9}
        assert by_id[9].offset[0] >= 23
        assert by_id[5].offset[0] <= 2

    def test_node_coords_land_on_matching_label_in_original_mask(
        self, two_objects_mask, full_config
    ):
        """Every node's global coordinates must fall on a pixel the original
        mask actually labeled with that node's own object_id - not some
        other object's, and not background."""
        result = analyze_segmentation_mask(two_objects_mask, full_config)
        assert result.node_records  # sanity: the fixture does produce nodes
        for record in result.node_records:
            idx = (record["coord_0"], record["coord_1"])
            assert two_objects_mask[idx] == record["object_id"]

    def test_object_graph_nodes_land_on_matching_label_in_original_mask(
        self, two_objects_mask, full_config
    ):
        """Same check at the graph level (not just the flattened node
        records): each ObjectResult.graph's own coordinates, offset into
        global space, must fall on that same object's label in the mask."""
        result = analyze_segmentation_mask(two_objects_mask, full_config)
        assert result.objects
        for obj in result.objects:
            offset = np.array(obj.offset)
            for coords in obj.graph.coordinates:
                idx = tuple((coords + offset).astype(int))
                assert two_objects_mask[idx] == obj.object_id

    def test_skeleton_matches_independent_per_object_processing(
        self, two_objects_mask, full_config
    ):
        combined = analyze_segmentation_mask(two_objects_mask, full_config)
        solo_5 = analyze_segmentation_mask(
            (two_objects_mask == 5).astype(np.uint8), full_config
        )
        solo_9 = analyze_segmentation_mask(
            (two_objects_mask == 9).astype(np.uint8), full_config
        )
        expected = np.maximum(solo_5.skeleton, solo_9.skeleton)
        assert np.array_equal(combined.skeleton, expected)

    def test_touching_objects_processed_as_separate_components(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 2:10] = 3
        mask[5:15, 10:18] = 4  # touches object 3 along column 10
        config = PipelineConfig(
            extraction=ExtractionConfig(summary=True), output=OutputConfig()
        )
        result = analyze_segmentation_mask(mask, config)
        assert {row["object_id"] for row in result.summary_features} == {3, 4}


def _straight_line_mask(shape=(10, 30), n_pixels=20):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5, 5 : 5 + n_pixels] = 1
    return mask


class TestSpacing:
    @pytest.fixture(autouse=True)
    def _reset_fractal_warning_flag(self):
        """The anisotropic-fractal-dimension warning is printed at most once
        per process (to avoid spamming stderr across a whole batch run) -
        reset that module-level flag around each test so tests that assert
        on the warning don't depend on test execution order."""
        import maskel.pipeline as pipeline_module

        pipeline_module._fractal_anisotropic_warned = False
        yield
        pipeline_module._fractal_anisotropic_warned = False

    def test_isotropic_spacing_scales_lengths(self):
        mask = _straight_line_mask()
        config_unit = PipelineConfig(
            extraction=ExtractionConfig(summary=True), output=OutputConfig()
        )
        config_scaled = PipelineConfig(
            extraction=ExtractionConfig(summary=True, spacing=(2.0, 2.0)),
            output=OutputConfig(),
        )

        result_unit = analyze_segmentation_mask(mask, config_unit)
        result_scaled = analyze_segmentation_mask(mask, config_scaled)

        unit_length = result_unit.summary_features[0]["total_length"]
        scaled_length = result_scaled.summary_features[0]["total_length"]
        assert scaled_length == pytest.approx(2.0 * unit_length)
        assert unit_length == pytest.approx(19.0)

    def test_anisotropic_spacing_on_axis_aligned_segment(self):
        mask = _straight_line_mask()
        config = PipelineConfig(
            extraction=ExtractionConfig(
                summary=True, mask_radius=True, spacing=(2.0, 0.5)
            ),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(mask, config)
        # line varies along axis 1 only: 19 unit steps * 0.5 spacing
        expected_length = 19 * 0.5
        assert result.summary_features[0]["total_length"] == pytest.approx(
            expected_length
        )

    def test_dimension_mismatch_warns_and_falls_back(self, capsys):
        mask = _straight_line_mask()  # 2D mask
        config = PipelineConfig(
            extraction=ExtractionConfig(summary=True, spacing=(1.0, 1.0, 1.0)),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(mask, config)
        captured = capsys.readouterr()
        assert "spacing" in captured.err.lower()
        assert result.summary_features  # ran to completion, didn't raise
        # falls back to isotropic: same as no spacing at all.
        config_none = PipelineConfig(
            extraction=ExtractionConfig(summary=True, spacing=None),
            output=OutputConfig(),
        )
        result_none = analyze_segmentation_mask(mask, config_none)
        assert result.summary_features[0]["total_length"] == pytest.approx(
            result_none.summary_features[0]["total_length"]
        )

    def test_fractal_dimension_anisotropic_spacing_forced_to_zero(self, capsys):
        vol = cross_volume()
        config = PipelineConfig(
            extraction=ExtractionConfig(
                summary=True, fractal_dimension=True, spacing=(1.0, 1.0, 2.0)
            ),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(vol, config)
        assert result.summary_features[0]["fractal_dimension"] == 0.0
        captured = capsys.readouterr()
        assert "fractal" in captured.err.lower()

    def test_fractal_dimension_isotropic_spacing_computes_normally(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(
                summary=True, fractal_dimension=True, spacing=(2.0, 2.0)
            ),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(cross_skel, config)
        assert result.summary_features[0]["fractal_dimension"] > 0

    def test_fractal_dimension_none_spacing_computes_normally(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(
                summary=True, fractal_dimension=True, spacing=None
            ),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(cross_skel, config)
        assert result.summary_features[0]["fractal_dimension"] > 0

    def test_fractal_dimension_not_requested_no_warning(self, capsys):
        vol = cross_volume()
        config = PipelineConfig(
            extraction=ExtractionConfig(
                summary=True, fractal_dimension=False, spacing=(1.0, 1.0, 2.0)
            ),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(vol, config)
        assert result.summary_features[0]["fractal_dimension"] == 0.0
        captured = capsys.readouterr()
        assert "fractal" not in captured.err.lower()


class TestProgressLogging:
    """analyze_segmentation_mask logs a few progress markers via the
    "maskel" logger (not print/stderr, unlike the warnings above) so
    impatient users get feedback without any logging setup of their own."""

    def _mask_with_two_objects(self):
        mask = np.zeros((16, 32), dtype=np.uint8)
        mask[4:8, 4:8] = 3
        mask[4:8, 20:24] = 7
        return mask

    def test_logs_shape_and_spacing(self, caplog):
        mask = self._mask_with_two_objects()
        config = PipelineConfig(
            extraction=ExtractionConfig(spacing=(1.0, 0.5)), output=OutputConfig()
        )
        with caplog.at_level("INFO", logger="maskel"):
            analyze_segmentation_mask(mask, config)

        messages = [r.message for r in caplog.records]
        assert any(
            "shape (16, 32)" in m and "spacing (1.0, 0.5)" in m for m in messages
        )

    def test_logs_object_count(self, caplog):
        mask = self._mask_with_two_objects()
        config = PipelineConfig(extraction=ExtractionConfig(), output=OutputConfig())
        with caplog.at_level("INFO", logger="maskel"):
            analyze_segmentation_mask(mask, config)

        messages = [r.message for r in caplog.records]
        assert any("Processing 2 object(s)" in m for m in messages)

    def test_logs_completion_with_timing_even_for_empty_mask(self, caplog):
        mask = np.zeros((8, 8), dtype=np.uint8)
        config = PipelineConfig(extraction=ExtractionConfig(), output=OutputConfig())
        with caplog.at_level("INFO", logger="maskel"):
            result = analyze_segmentation_mask(mask, config)

        assert result.objects == []
        messages = [r.message for r in caplog.records]
        assert any(m.startswith("Finished image of shape (8, 8) in") for m in messages)

    def test_dimension_mismatch_warning_still_goes_to_stderr_not_the_logger(
        self, caplog, capsys
    ):
        mask = self._mask_with_two_objects()
        config = PipelineConfig(
            extraction=ExtractionConfig(spacing=(1.0, 1.0, 1.0)),
            output=OutputConfig(),
        )
        with caplog.at_level("INFO", logger="maskel"):
            analyze_segmentation_mask(mask, config)

        assert not any("Warning:" in r.message for r in caplog.records)
        assert "spacing" in capsys.readouterr().err.lower()
