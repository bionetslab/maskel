"""Unit tests for the 2D thinning kernel's input validation.

Correctness against scikit-image's own Lee thinning is covered separately in
tests/test_2d_skimage_comparison.py - this file covers thin_2d's own guards.
"""

import numpy as np
import pytest

from maskel.thin_2d import thin_2d


class TestBinaryGuard:
    """thin_2d's kernels test foreground with ``== 1``, so non-binary input
    has to be refused rather than silently surviving the peeling passes -
    mirrors thin_3d's TestBinaryGuard in tests/test_thin_3d.py."""

    @pytest.mark.parametrize(
        "img",
        [
            pytest.param(np.array([[0, 7, 300]], dtype=np.uint16), id="grayscale"),
            pytest.param(np.array([[0, 255]], dtype=np.uint8), id="0-255-mask"),
            pytest.param(np.array([[0, 2]], dtype=np.uint8), id="label-map"),
            pytest.param(np.array([[-1, 1]], dtype=np.int8), id="negative"),
        ],
    )
    def test_rejects_values_outside_zero_and_one(self, img):
        with pytest.raises(ValueError, match="must be binary"):
            thin_2d(img)

    def test_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match="Expected 2D input"):
            thin_2d(np.zeros((2, 2, 2), dtype=np.uint8))


class TestUniformInput:
    """An image that's entirely background or entirely foreground has no
    boundary to erode from - it short-circuits to an empty result instead
    of tripping the strict binary-range check, which would otherwise reject
    a perfectly valid (if degenerate) binary array just because min()==max().
    """

    def test_all_background_returns_empty(self):
        img = np.zeros((6, 6), dtype=np.uint8)
        result = thin_2d(img)
        assert result.shape == img.shape
        assert not result.any()

    def test_all_foreground_returns_empty(self):
        img = np.ones((6, 6), dtype=np.uint8)
        result = thin_2d(img)
        assert result.shape == img.shape
        assert not result.any()

    @pytest.mark.parametrize("value", [7, 255])
    def test_uniform_non_binary_value_still_rejected(self, value):
        # A uniform array isn't automatically "empty either way" - only 0
        # and 1 are, so this must still hit the binary-range error.
        img = np.full((3, 3), value, dtype=np.uint8)
        with pytest.raises(ValueError, match="must be binary"):
            thin_2d(img)
