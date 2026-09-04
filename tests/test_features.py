import numpy as np
import pytest
from skan import Skeleton, summarize

from maskel.features import (
    _EMPTY_FEATURES,
    build_skeleton_graph,
    compute_radii,
    compute_tortuosity,
    extract_node_features,
    extract_summary_features,
    fractal_dimension,
    per_segment_radii,
)


class TestComputeTortuosity:
    def test_basic(self):
        lengths = np.array([10.0, 5.0, 20.0])
        euclidean = np.array([5.0, 5.0, 10.0])
        result = compute_tortuosity(lengths, euclidean)
        np.testing.assert_allclose(result, [2.0, 1.0, 2.0])

    def test_zero_euclidean_returns_nan(self):
        lengths = np.array([1.0, 2.0, 3.0])
        euclidean = np.array([0.0, 5.0, 0.0])
        result = compute_tortuosity(lengths, euclidean)
        assert np.isnan(result[0])
        assert result[1] == 0.4
        assert np.isnan(result[2])

    def test_empty_arrays(self):
        result = compute_tortuosity(np.array([]), np.array([]))
        assert len(result) == 0

    def test_output_dtype_is_float(self):
        result = compute_tortuosity(np.ones(5), np.ones(5))
        assert result.dtype == np.float64


class TestFractalDimension:
    def test_straight_2d_line(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        img[32, :] = 1
        fd, r2 = fractal_dimension(img)
        assert fd > 0.8
        assert fd < 1.2
        assert r2 > 0.9

    def test_filled_2d_square(self):
        img = np.ones((64, 64), dtype=np.uint8)
        fd, _ = fractal_dimension(img)
        assert fd > 1.5

    def test_straight_3d_line(self):
        vol = np.zeros((32, 32, 32), dtype=np.uint8)
        vol[16, 16, :] = 1
        fd, r2 = fractal_dimension(vol)
        assert fd > 0.8
        assert fd < 1.3
        assert r2 > 0.9

    def test_empty_image(self):
        fd, r2 = fractal_dimension(np.zeros((64, 64), dtype=np.uint8))
        assert fd == 0.0
        assert r2 == 0.0

    def test_minimum_size_returns_zero(self):
        fd, r2 = fractal_dimension(np.ones((5, 5), dtype=np.uint8))
        assert fd == 0.0
        assert r2 == 0.0

    def test_32x32_line_works(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, :] = 1
        fd, r2 = fractal_dimension(img)
        assert fd > 0.8
        assert fd < 1.2
        assert r2 > 0.9


class TestComputeRadii:
    def test_radius_and_diameter_stats(self):
        binary = np.ones((7, 7), dtype=np.uint8)
        binary[0, :] = binary[-1, :] = binary[:, 0] = binary[:, -1] = 0
        skeleton = np.zeros((7, 7), dtype=np.uint8)
        skeleton[3, 3] = 1

        radius_matrix, stats = compute_radii(binary, skeleton)
        assert stats["mean_radius"] == 3
        assert radius_matrix.shape == binary.shape
        assert stats["mean_diameter"] == stats["mean_radius"] * 2.0
        assert stats["min_diameter"] == stats["min_radius"] * 2.0
        assert stats["max_diameter"] == stats["max_radius"] * 2.0
        assert stats["std_diameter"] == stats["std_radius"] * 2.0

    def test_empty_skeleton(self):
        binary = np.ones((10, 10), dtype=np.uint8)
        binary[0, 0] = 0
        skeleton = np.zeros((10, 10), dtype=np.uint8)
        radius_matrix, stats = compute_radii(binary, skeleton)
        assert stats["mean_radius"] == 0.0
        assert stats["mean_diameter"] == 0.0
        assert not radius_matrix.any()

    def test_3d_radius(self):
        vol = np.ones((16, 16, 16), dtype=np.uint8)
        skel = np.zeros_like(vol)
        skel[8, 8, 8] = 1
        radius_matrix, stats = compute_radii(vol, skel)
        assert stats["mean_radius"] > 0
        assert radius_matrix.shape == vol.shape

    def test_radius_matrix_zeros_outside_skeleton(self):
        binary = np.ones((5, 5), dtype=np.uint8)
        skeleton = np.zeros((5, 5), dtype=np.uint8)
        skeleton[2, 2] = 1
        radius_matrix, _ = compute_radii(binary, skeleton)
        assert radius_matrix[2, 2] > 0
        assert radius_matrix[0, 0] == 0
        assert radius_matrix.dtype == np.float64

    def test_anisotropic_spacing_scales_axis_aligned_distance(self):
        # Rows 0 and 6 are background; rows 1-5 and every column are
        # foreground, so the strip is unbounded along columns - the
        # nearest background pixel to any interior point is directly
        # above/below in the same column, making the expected physical
        # distance exactly row_distance * row_spacing regardless of the
        # column spacing.
        binary = np.ones((7, 20), dtype=np.uint8)
        binary[0, :] = 0
        binary[6, :] = 0
        skeleton = np.zeros((7, 20), dtype=np.uint8)
        skeleton[3, 10] = 1

        radius_matrix, stats = compute_radii(binary, skeleton, spacing=(1.5, 0.5))
        assert radius_matrix[3, 10] == pytest.approx(3 * 1.5)
        assert stats["mean_radius"] == pytest.approx(4.5)

    def test_none_spacing_matches_isotropic_unit_spacing(self):
        binary = np.ones((7, 7), dtype=np.uint8)
        binary[0, :] = binary[-1, :] = binary[:, 0] = binary[:, -1] = 0
        skeleton = np.zeros((7, 7), dtype=np.uint8)
        skeleton[3, 3] = 1

        rm_none, stats_none = compute_radii(binary, skeleton, spacing=None)
        rm_unit, stats_unit = compute_radii(binary, skeleton, spacing=(1.0, 1.0))
        np.testing.assert_allclose(rm_none, rm_unit)
        assert stats_none == stats_unit


class TestBuildVesselGraph:
    def test_returns_skeleton_instance(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        img[4, 2:6] = 1
        graph = build_skeleton_graph(img)
        assert isinstance(graph, Skeleton)

    def test_isotropic_spacing_scales_branch_length(self):
        # 20-pixel straight horizontal line: 19 unit steps.
        img = np.zeros((10, 30), dtype=np.uint8)
        img[5, 5:25] = 1

        graph_unit = build_skeleton_graph(img)
        data_unit = summarize(graph_unit, separator="-")

        graph_scaled = build_skeleton_graph(img, spacing=(2.0, 2.0))
        data_scaled = summarize(graph_scaled, separator="-")

        expected_unit_length = 19.0
        assert data_unit["branch-distance"].iloc[0] == pytest.approx(
            expected_unit_length
        )
        assert data_scaled["branch-distance"].iloc[0] == pytest.approx(
            2.0 * expected_unit_length
        )
        # spacing doubles both distances identically, so tortuosity is
        # unaffected.
        assert data_scaled["euclidean-distance"].iloc[0] == pytest.approx(
            2.0 * data_unit["euclidean-distance"].iloc[0]
        )

    def test_anisotropic_spacing_scales_axis_aligned_segment(self):
        # 20-pixel straight horizontal line (varies along axis 1 only):
        # 19 unit steps, so with spacing=(2.0, 0.5) each step is 0.5.
        img = np.zeros((10, 30), dtype=np.uint8)
        img[5, 5:25] = 1

        graph = build_skeleton_graph(img, spacing=(2.0, 0.5))
        data = summarize(graph, separator="-")

        expected_length = 19 * 0.5
        assert data["branch-distance"].iloc[0] == pytest.approx(expected_length)
        assert data["euclidean-distance"].iloc[0] == pytest.approx(expected_length)

    def test_coordinates_unaffected_by_spacing(self):
        img = np.zeros((8, 8), dtype=np.uint8)
        img[4, 2:6] = 1
        graph_unit = build_skeleton_graph(img)
        graph_scaled = build_skeleton_graph(img, spacing=(3.0, 3.0))
        np.testing.assert_array_equal(graph_unit.coordinates, graph_scaled.coordinates)


class TestExtractVesselFeatures:
    @pytest.fixture
    def simple_cross(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 8:24] = 1
        img[8:24, 16] = 1
        return img

    @pytest.fixture
    def simple_cross_graph(self, simple_cross):
        return build_skeleton_graph(simple_cross)

    @pytest.fixture
    def simple_cross_branch_data(self, simple_cross_graph):
        return summarize(simple_cross_graph, separator="-")

    @pytest.fixture
    def cross_features(
        self, simple_cross, simple_cross_graph, simple_cross_branch_data
    ):
        return extract_summary_features(
            simple_cross,
            simple_cross_graph,
            simple_cross_branch_data,
            binary=simple_cross,
        )

    def test_cross_shape_topology(self, cross_features):
        assert cross_features["num_endpoints"] == 4
        assert cross_features["num_bifurcations"] == 1

    def test_all_keys_present(self, cross_features):
        assert set(cross_features.keys()) == set(_EMPTY_FEATURES.keys())

    def test_defaults_radius_to_zero(self, cross_features):
        assert cross_features["mean_radius"] == 0.0
        assert cross_features["mean_diameter"] == 0.0

    @pytest.mark.parametrize("include_fractal", [True, False])
    def test_fractal_dimension_flag(
        self,
        simple_cross,
        simple_cross_graph,
        simple_cross_branch_data,
        include_fractal,
    ):
        features = extract_summary_features(
            simple_cross,
            simple_cross_graph,
            simple_cross_branch_data,
            binary=simple_cross,
            include_fractal=include_fractal,
        )
        if include_fractal:
            assert features["fractal_dimension"] > 0
            assert features["fractal_dimension_r2"] > 0
        else:
            assert features["fractal_dimension"] == 0.0
            assert features["fractal_dimension_r2"] == 0.0

    def test_radius_stats_passthrough(
        self, simple_cross, simple_cross_graph, simple_cross_branch_data
    ):
        radius_stats = {
            "mean_radius": 3.5,
            "std_radius": 0.5,
            "min_radius": 2.0,
            "max_radius": 5.0,
            "mean_diameter": 7.0,
            "std_diameter": 1.0,
            "min_diameter": 4.0,
            "max_diameter": 10.0,
            "mean_segment_volume": 0.0,
            "mean_surface_area": 0.0,
        }
        features = extract_summary_features(
            simple_cross,
            simple_cross_graph,
            simple_cross_branch_data,
            binary=simple_cross,
            radius_stats=radius_stats,
        )
        assert features["mean_radius"] == 3.5
        assert features["max_radius"] == 5.0

    def test_single_straight_line(self):
        img = np.zeros((8, 32), dtype=np.uint8)
        img[4, 4:28] = 1
        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-")
        features = extract_summary_features(
            img, graph, branch_data, binary=img, include_fractal=False
        )
        assert features["num_nodes"] == 2.0
        assert features["num_endpoints"] == 2.0
        assert features["total_length"] > 0

    def test_empty_branch_data_returns_zeroes(
        self, simple_cross, simple_cross_graph, simple_cross_branch_data
    ):
        empty_data = simple_cross_branch_data.iloc[0:0]
        features = extract_summary_features(
            simple_cross, simple_cross_graph, empty_data, binary=simple_cross
        )
        for v in features.values():
            assert v == 0.0

    def test_hgu_is_total_length_over_num_endpoints(self):
        img = np.zeros((8, 32), dtype=np.uint8)
        img[4, 4:28] = 1
        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-")
        features = extract_summary_features(
            img, graph, branch_data, binary=img, include_fractal=False
        )
        expected_hgu = features["total_length"] / features["num_endpoints"]
        assert features["hgu"] == expected_hgu

    def test_length_statistics_are_consistent(self, cross_features):
        assert (
            cross_features["min_length"]
            <= cross_features["mean_length"]
            <= cross_features["max_length"]
        )

    def test_vessel_area_from_binary(
        self, simple_cross, simple_cross_graph, simple_cross_branch_data
    ):
        features = extract_summary_features(
            simple_cross,
            simple_cross_graph,
            simple_cross_branch_data,
            binary=simple_cross,
        )
        assert features["vessel_area"] == np.count_nonzero(simple_cross)
        assert (
            features["vessel_area_fraction"]
            == np.count_nonzero(simple_cross) / simple_cross.size
        )

    def test_vessel_area_scaled_by_spacing(
        self, simple_cross, simple_cross_graph, simple_cross_branch_data
    ):
        features = extract_summary_features(
            simple_cross,
            simple_cross_graph,
            simple_cross_branch_data,
            binary=simple_cross,
            spacing=(2.0, 0.5),
        )
        expected_area = np.count_nonzero(simple_cross) * 2.0 * 0.5
        assert features["vessel_area"] == pytest.approx(expected_area)
        # vessel_area_fraction is a ratio - scaling cancels, no change.
        assert (
            features["vessel_area_fraction"]
            == np.count_nonzero(simple_cross) / simple_cross.size
        )


class TestExtractNodeFeatures:
    @pytest.fixture
    def simple_cross(self):
        img = np.zeros((32, 32), dtype=np.uint8)
        img[16, 8:24] = 1
        img[8:24, 16] = 1
        return img

    @pytest.fixture
    def simple_cross_graph(self, simple_cross):
        return build_skeleton_graph(simple_cross)

    @pytest.fixture
    def simple_cross_branch_data(self, simple_cross_graph):
        return summarize(simple_cross_graph, separator="-")

    def test_returns_one_record_per_graph_node(
        self, simple_cross_graph, simple_cross_branch_data
    ):
        nodes = extract_node_features(simple_cross_graph, simple_cross_branch_data)
        assert len(nodes) == simple_cross_graph.coordinates.shape[0]

    def test_has_expected_keys(self, simple_cross_graph, simple_cross_branch_data):
        nodes = extract_node_features(simple_cross_graph, simple_cross_branch_data)
        assert set(nodes[0].keys()) == {
            "node_id",
            "degree",
            "coord_0",
            "coord_1",
            "is_endpoint",
            "is_junction",
            "is_pass_through",
        }

    def test_cross_topology_counts(self, simple_cross_graph, simple_cross_branch_data):
        nodes = extract_node_features(simple_cross_graph, simple_cross_branch_data)
        endpoints = sum(1 for n in nodes if n["is_endpoint"])
        junctions = sum(1 for n in nodes if n["is_junction"])
        assert endpoints == 4
        assert junctions == 1

    def test_radius_included_when_provided(
        self, simple_cross, simple_cross_graph, simple_cross_branch_data
    ):
        radius_matrix, _ = compute_radii(
            simple_cross, (simple_cross > 0).astype(np.uint8)
        )
        nodes = extract_node_features(
            simple_cross_graph, simple_cross_branch_data, radius_matrix=radius_matrix
        )
        assert "radius" in nodes[0]
        assert all(
            n["radius"] > 0 for n in nodes if n["is_endpoint"] or n["is_junction"]
        )

    def test_no_radius_when_not_provided(
        self, simple_cross_graph, simple_cross_branch_data
    ):
        nodes = extract_node_features(simple_cross_graph, simple_cross_branch_data)
        assert "radius" not in nodes[0]

    def test_nodes_have_pixel_coordinates(
        self, simple_cross_graph, simple_cross_branch_data
    ):
        nodes = extract_node_features(simple_cross_graph, simple_cross_branch_data)
        coords = np.array([(n["coord_0"], n["coord_1"]) for n in nodes])
        assert coords.shape == (len(nodes), 2)
        assert coords.dtype == np.int64

    def test_works_with_single_straight_line(self):
        img = np.zeros((8, 32), dtype=np.uint8)
        img[4, 4:28] = 1
        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-")
        nodes = extract_node_features(graph, branch_data)
        assert len(nodes) > 0
        assert nodes[0]["is_endpoint"]
        assert nodes[-1]["is_endpoint"]
        assert all(n["is_pass_through"] or n["is_endpoint"] for n in nodes)


class TestPerSegmentRadii:
    """Cylinder-formula checks for the frustum-chain volume/surface_area fix."""

    def _straight_line_graph(self, n_pixels=20):
        img = np.zeros((10, 30), dtype=np.uint8)
        img[5, 5 : 5 + n_pixels] = 1
        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-")
        return img, graph, branch_data

    def _diagonal_line_graph(self, n_pixels=20):
        img = np.zeros((30, 30), dtype=np.uint8)
        for i in range(n_pixels):
            img[5 + i, 5 + i] = 1
        graph = build_skeleton_graph(img)
        branch_data = summarize(graph, separator="-")
        return img, graph, branch_data

    def test_straight_line_cylinder_isotropic(self):
        img, graph, branch_data = self._straight_line_graph(n_pixels=20)
        radius_matrix = np.zeros_like(img, dtype=np.float64)
        radius_matrix[img > 0] = 3.0
        n = len(branch_data)

        result = per_segment_radii(radius_matrix, graph, n)

        expected_length = 19.0  # 20 pixels -> 19 unit steps
        r = 3.0
        expected_volume = np.pi * r**2 * expected_length
        expected_surface = 2.0 * np.pi * r * expected_length
        assert result["volume"][0] == pytest.approx(expected_volume)
        assert result["surface_area"][0] == pytest.approx(expected_surface)

    def test_straight_line_cylinder_anisotropic(self):
        img, graph, branch_data = self._straight_line_graph(n_pixels=20)
        radius_matrix = np.zeros_like(img, dtype=np.float64)
        radius_matrix[img > 0] = 3.0
        n = len(branch_data)
        spacing = (2.0, 0.5)  # line varies along axis 1 only

        result = per_segment_radii(radius_matrix, graph, n, spacing=spacing)

        expected_length = 19 * 0.5
        r = 3.0
        expected_volume = np.pi * r**2 * expected_length
        expected_surface = 2.0 * np.pi * r * expected_length
        assert result["volume"][0] == pytest.approx(expected_volume)
        assert result["surface_area"][0] == pytest.approx(expected_surface)

    def test_diagonal_line_cylinder_isotropic(self):
        img, graph, branch_data = self._diagonal_line_graph(n_pixels=20)
        radius_matrix = np.zeros_like(img, dtype=np.float64)
        radius_matrix[img > 0] = 2.0
        n = len(branch_data)

        result = per_segment_radii(radius_matrix, graph, n)

        expected_length = 19 * np.sqrt(2.0)  # 19 diagonal unit steps
        r = 2.0
        expected_volume = np.pi * r**2 * expected_length
        expected_surface = 2.0 * np.pi * r * expected_length
        assert result["volume"][0] == pytest.approx(expected_volume)
        assert result["surface_area"][0] == pytest.approx(expected_surface)

    def test_diagonal_line_cylinder_anisotropic(self):
        img, graph, branch_data = self._diagonal_line_graph(n_pixels=20)
        radius_matrix = np.zeros_like(img, dtype=np.float64)
        radius_matrix[img > 0] = 2.0
        n = len(branch_data)
        spacing = (2.0, 0.5)

        result = per_segment_radii(radius_matrix, graph, n, spacing=spacing)

        step_length = np.sqrt((1 * 2.0) ** 2 + (1 * 0.5) ** 2)
        expected_length = 19 * step_length
        r = 2.0
        expected_volume = np.pi * r**2 * expected_length
        expected_surface = 2.0 * np.pi * r * expected_length
        assert result["volume"][0] == pytest.approx(expected_volume)
        assert result["surface_area"][0] == pytest.approx(expected_surface)

    def test_empty_branch_yields_nan(self):
        img, graph, branch_data = self._straight_line_graph(n_pixels=20)
        radius_matrix = np.zeros_like(img, dtype=np.float64)  # no positive radii
        n = len(branch_data)

        result = per_segment_radii(radius_matrix, graph, n)

        assert np.isnan(result["volume"][0])
        assert np.isnan(result["surface_area"][0])
        assert np.isnan(result["mean_radius"][0])
