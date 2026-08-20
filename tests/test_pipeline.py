import numpy as np
import pytest

from maskel.config import ExtractionConfig, OutputConfig, PipelineConfig
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
