"""Regression test comparing maskel.thin with scikit-image skeletonize on a 2D image.

A version of this test existed before the vesskel -> maskel rename/repo-split
(commit e0910f0) but depended on the HRF dataset, which moved to the separate
maskel-evaluations repo along with it - this is a from-scratch replacement using
a bundled scikit-image sample image instead, so it needs no external data and can
run in CI the same way tests/test_3d_skimage_comparison.py already does.
"""

import numpy as np
import pytest
from skimage import data
from skimage.morphology import skeletonize

from maskel.thin import lee94_thin


@pytest.mark.slow
def test_maskel_vs_scikit_skeletonize_2d():
    # data.horse() is already a 2D boolean silhouette, so no thresholding
    # is needed - just an inversion, since it's True for the (majority)
    # background and False for the horse shape itself.
    image = (~data.horse()).astype(np.uint8)

    maskel_skel = lee94_thin(image)
    # skimage.morphology.skeletonize defaults to Zhang's algorithm for 2D
    # input (Lee only for 3D) - method="lee" is required here for this to
    # be a comparison against the same algorithm lee94_thin implements.
    scikit_skel = skeletonize(image, method="lee")

    assert maskel_skel.shape == scikit_skel.shape, (
        f"shape mismatch: maskel {maskel_skel.shape} vs scikit {scikit_skel.shape}"
    )
    assert np.array_equal(maskel_skel, scikit_skel), (
        "skeleton mismatch: algorithms produce different results"
    )
