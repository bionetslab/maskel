import numpy as np
import pytest

from maskel.config import ExtractionConfig, OutputConfig, PipelineConfig
from maskel.pipeline import AnalysisResult, analyze_binary_image

from ._helpers import cross_volume, line_volume


class TestAnalysisResult:
    def test_default_radius_matrix_is_none(self):
        result = AnalysisResult(
            skeleton=np.zeros((2, 2), dtype=np.uint8),
            summary_features={},
            branch_records=[],
            node_records=[],
        )
        assert result.radius_matrix is None

    def test_construct_with_all_fields(self):
        skeleton = np.eye(4, dtype=np.uint8)
        radius = np.ones_like(skeleton, dtype=np.float64)
        result = AnalysisResult(
            skeleton=skeleton,
            summary_features={"num_nodes": 3.0},
            branch_records=[{"branch_id": 0}],
            node_records=[],
            radius_matrix=radius,
        )
        assert np.array_equal(result.skeleton, skeleton)
        assert result.summary_features == {"num_nodes": 3.0}
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
        assert result.summary_features == {}
        assert result.branch_records == []
        assert result.radius_matrix is None
        assert result.graph is None
        assert result.branch_data is None

    def test_result_carries_graph_and_branch_data(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
        assert result.graph is not None
        assert result.branch_data is not None
        assert not result.branch_data.empty

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
        assert result.summary_features == {}

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
        assert result.summary_features["mean_radius"] > 0

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
        for key in radius_keys:
            assert key in result.summary_features
            assert result.summary_features[key] > 0

    def test_fractal_dimension_disabled_by_default(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
        assert result.summary_features["fractal_dimension"] == 0.0

    def test_fractal_dimension_enabled(self, cross_skel):
        config = PipelineConfig(
            extraction=ExtractionConfig(fractal_dimension=True, summary=True),
            output=OutputConfig(),
        )
        result = analyze_binary_image(cross_skel, config)
        assert result.summary_features["fractal_dimension"] > 0

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
        assert result.summary_features["fractal_dimension"] > 0
        assert result.summary_features["mean_radius"] > 0

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
        assert result.summary_features == {}
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
        assert result.summary_features["mean_radius"] > 0

    def test_cross_topology_num_endpoints(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
        assert result.summary_features["num_endpoints"] == 4

    def test_cross_topology_num_bifurcations(self, cross_skel, analysis_config):
        result = analyze_binary_image(cross_skel, analysis_config)
        assert result.summary_features["num_bifurcations"] == 1
