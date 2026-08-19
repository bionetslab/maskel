"""Regression test for 3D thinning using brain image from scikit-image."""

import numpy as np
import pytest
from skan import summarize
from skimage import data

from maskel.features import build_vessel_graph, compute_radii, extract_vessel_features
from maskel.thin import lee94_thin

from ._helpers import (
    BASELINE_DIR,
    FEATURE_DIR,
    feature_path,
    hash_array,
    read_feature_csv,
    skeleton_path,
    write_feature_csv,
)


def _compute_skeleton(image: np.ndarray) -> np.ndarray:
    return lee94_thin(image)


@pytest.mark.slow
class Test3DThinningRegression:
    """Run 3D thinning on brain image and compare against saved baselines."""

    @pytest.fixture(scope="class")
    def image(self):
        return data.brain()

    def test_skeleton_matches_baseline(self, image, request):
        skeleton = _compute_skeleton(image)
        graph = build_vessel_graph(skeleton)
        branch_data = summarize(graph, separator="-")
        _, radius_stats = compute_radii(image, skeleton)
        features = extract_vessel_features(
            skeleton,
            graph,
            branch_data,
            binary=image,
            radius_stats=radius_stats,
        )
        name = "brain"
        baseline_file = skeleton_path(name)
        feature_file = feature_path(name)

        baseline_changed = False
        if request.config.getoption("--update-baseline") or not baseline_file.exists():
            BASELINE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(baseline_file, skeleton=skeleton)
            baseline_changed = True

        features_changed = False
        if request.config.getoption("--update-baseline") or not feature_file.exists():
            FEATURE_DIR.mkdir(parents=True, exist_ok=True)
            write_feature_csv(feature_file, features)
            features_changed = True

        if baseline_changed or features_changed:
            artifacts = []
            if baseline_changed:
                artifacts.append("skeleton baseline")
            if features_changed:
                artifacts.append("feature baseline")
            pytest.skip(f"{', '.join(artifacts)} created for baseline")

        with np.load(baseline_file) as data:
            baseline = data["skeleton"]

        assert skeleton.shape == baseline.shape, (
            f"shape mismatch got {skeleton.shape}, expected {baseline.shape}"
        )
        assert np.array_equal(skeleton, baseline), (
            f"skeleton differs (hash {hash_array(skeleton)} vs {hash_array(baseline)})"
        )

        baseline_features = read_feature_csv(feature_file)
        feature_keys = sorted(features)
        baseline_feature_keys = sorted(baseline_features)
        assert feature_keys == baseline_feature_keys, (
            f"feature set differs (got {feature_keys}, expected {baseline_feature_keys})"
        )

        feature_values = np.array([features[k] for k in feature_keys], dtype=np.float64)
        baseline_feature_values = np.array(
            [baseline_features[k] for k in feature_keys], dtype=np.float64
        )
        np.testing.assert_allclose(
            feature_values,
            baseline_feature_values,
            rtol=1e-8,
            atol=1e-10,
            err_msg="feature values differ from baseline",
        )
