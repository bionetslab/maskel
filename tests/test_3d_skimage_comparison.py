"""Regression test comparing maskel.thin with scikit-image skeletonize on brain image."""

import numpy as np
import pytest
from skimage import data
from skimage.morphology import skeletonize

from maskel.thin import lee94_thin


@pytest.mark.slow
class TestSkeletonizeComparison:
    """Compare maskel.thin with scikit-image skeletonize on brain image."""

    @pytest.fixture(scope="class")
    def image(self):
        return data.brain()

    def test_maskel_vs_scikit_skeletonize(self, image):
        maskel_skel = lee94_thin(image)
        scikit_skel = skeletonize(image)

        assert maskel_skel.shape == scikit_skel.shape, (
            f"shape mismatch: maskel {maskel_skel.shape} vs scikit {scikit_skel.shape}"
        )
        assert np.array_equal(maskel_skel, scikit_skel), (
            "skeleton mismatch: algorithms produce different results"
        )
