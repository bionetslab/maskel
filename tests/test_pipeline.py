import numpy as np
import pytest

from maskel.config import ExtractionConfig, OutputConfig, PipelineConfig
from maskel.pipeline import AnalysisResult, analyze_binary_image

from ._helpers import cross_volume, line_volume


class TestAnalysisResult:
    def test_default_radius_matrix_is_none(self):
        result = AnalysisResult(
            skeleton=np.zeros((2, 2), dtype=np.uint8),
            summary_features=[],
            branch_records=[],
            node_records=[],
        )
        assert result.radius_matrix is None

    def test_construct_with_all_fields(self):
        skeleton = np.eye(4, dtype=np.uint8)
        radius = np.ones_like(skeleton, dtype=np.float64)
        result = AnalysisResult(
            skeleton=skeleton,
            summary_features=[{"num_nodes": 3.0, "object_id": 1}],
            branch_records=[{"branch_id": 0}],
            node_records=[],
            radius_matrix=radius,
        )
        assert np.array_equal(result.skeleton, skeleton)
        assert result.summary_features == [{"num_nodes": 3.0, "object_id": 1}]
        assert result.branch_records == [{"branch_id": 0}]
        assert np.array_equal(result.radius_matrix, radius)


class TestAnalyzeBinaryImage:
    @pytest.fixture
    def analysis_config(self):
        """Shared config with branches and summary enabled."""
        return PipelineConfig(
            extraction=ExtractionConfig(branches=True, summary=True),
            output=OutputConfig(),
        )

    def test_empty_image_returns_empty_result(self, analysis_config):
        img = np.zeros((32, 32), dtype=np.uint8)
        result = analyze_binary_image(img, analysis_config)
        assert not result.skeleton.any()
        assert result.summary_features == []
        assert result.branch_records == []
        assert result.radius_matrix is None
        assert result.object_graphs == []

    def test_result_carries_object_graphs(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
        assert len(result.object_graphs) == 1
        og = result.object_graphs[0]
        assert og.object_id == 1
        assert not og.branch_data.empty

    def test_cross_produces_skeleton(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
        assert result.skeleton.any()
        assert result.skeleton.dtype == np.uint8
        assert result.skeleton.shape == cross_skel.shape

    def test_non_binary_input_is_binarized(self, analysis_config):
        img = np.zeros((20, 20), dtype=np.int32)
        img[10, 5:15] = 200
        result = analyze_binary_image(img, analysis_config)
        assert result.skeleton.any()
        assert set(np.unique(result.skeleton)) <= {0, 1}

    def test_summary_disabled_returns_empty_features(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(summary=False),
            output=OutputConfig(),
        )
        result = analyze_binary_image(cross_skel, config)
        assert result.summary_features == []

    def test_binary_input_gets_object_id_one(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
        assert result.summary_features[0]["object_id"] == 1
        assert all(r["object_id"] == 1 for r in result.branch_records)

    def test_branches_enabled(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
        assert len(result.branch_records) > 0
        assert isinstance(result.branch_records[0], dict)

    def test_branches_disabled_returns_empty_records(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(branches=False),
            output=OutputConfig(),
        )
        result = analyze_binary_image(cross_skel, config)
        assert result.branch_records == []

    def test_vessel_radius_enabled(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(vessel_radius=True, summary=True),
            output=OutputConfig(),
        )
        result = analyze_binary_image(cross_skel, config)
        assert result.radius_matrix is not None
        assert result.radius_matrix.shape == cross_skel.shape
        assert result.radius_matrix.any()
        assert result.summary_features[0]["mean_radius"] > 0

    def test_vessel_radius_disabled_radius_none(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
        assert result.radius_matrix is None

    def test_radius_stats_in_summary_when_enabled(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(vessel_radius=True, summary=True),
            output=OutputConfig(),
        )
        result = analyze_binary_image(cross_skel, config)
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
        result = analyze_binary_image(cross_skel, analysis_config)
        assert result.summary_features[0]["fractal_dimension"] == 0.0

    def test_fractal_dimension_enabled(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(fractal_dimension=True, summary=True),
            output=OutputConfig(),
        )
        result = analyze_binary_image(cross_skel, config)
        assert result.summary_features[0]["fractal_dimension"] > 0

    def test_all_options_enabled(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(
                branches=True,
                branch_text=True,
                summary=True,
                fractal_dimension=True,
                vessel_radius=True,
            ),
            output=OutputConfig(),
        )
        result = analyze_binary_image(cross_skel, config)
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
                vessel_radius=False,
            ),
            output=OutputConfig(),
        )
        result = analyze_binary_image(cross_skel, config)
        assert result.skeleton.any()
        assert result.radius_matrix is None
        assert result.summary_features == []
        assert result.branch_records == []

    def test_3d_image(self):
        vol = cross_volume()
        config = PipelineConfig(
            extraction=ExtractionConfig(summary=True), output=OutputConfig()
        )
        result = analyze_binary_image(vol, config)
        assert result.skeleton.any()
        assert len(result.summary_features) > 0

    def test_3d_image_with_radius(self):
        vol = line_volume((12, 12, 12), axis=2)
        config = PipelineConfig(
            extraction=ExtractionConfig(
                vessel_radius=True, fractal_dimension=True, summary=True
            ),
            output=OutputConfig(),
        )
        result = analyze_binary_image(vol, config)
        assert result.radius_matrix is not None
        assert result.radius_matrix.shape == vol.shape
        assert result.summary_features[0]["mean_radius"] > 0

    def test_cross_topology_num_endpoints(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
        assert result.summary_features[0]["num_endpoints"] == 4

    def test_cross_topology_num_bifurcations(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
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
        result = analyze_binary_image(two_objects_mask, full_config)
        assert len(result.summary_features) == 2
        assert {row["object_id"] for row in result.summary_features} == {5, 9}

    def test_object_id_matches_mask_label_not_renumbered(
        self, two_objects_mask, full_config
    ):
        result = analyze_binary_image(two_objects_mask, full_config)
        assert {r["object_id"] for r in result.branch_records} == {5, 9}
        assert {r["object_id"] for r in result.node_records} == {5, 9}

    def test_node_coords_are_in_global_image_frame(self, two_objects_mask, full_config):
        result = analyze_binary_image(two_objects_mask, full_config)
        object_9_rows = [r for r in result.node_records if r["object_id"] == 9]
        assert object_9_rows
        for r in object_9_rows:
            assert r["coord_0"] >= 23
            assert r["coord_1"] >= 23

    def test_object_graphs_tagged_and_offset(self, two_objects_mask, full_config):
        result = analyze_binary_image(two_objects_mask, full_config)
        by_id = {og.object_id: og for og in result.object_graphs}
        assert set(by_id) == {5, 9}
        assert by_id[9].offset[0] >= 23
        assert by_id[5].offset[0] <= 2

    def test_skeleton_matches_independent_per_object_processing(
        self, two_objects_mask, full_config
    ):
        combined = analyze_binary_image(two_objects_mask, full_config)
        solo_5 = analyze_binary_image(
            (two_objects_mask == 5).astype(np.uint8), full_config
        )
        solo_9 = analyze_binary_image(
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
        result = analyze_binary_image(mask, config)
        assert {row["object_id"] for row in result.summary_features} == {3, 4}
