import numpy as np
import pytest
from scipy import ndimage as ndi

from maskel._utils import _label_compact
from maskel.config import PreprocessingConfig
from maskel.preprocessing import preprocess_binary, preprocessing_margin


def _ring(size: int, hole: tuple[int, int]) -> np.ndarray:
    """Square ring with a background border and a *hole*-sized hole in the middle."""
    img = np.zeros((size, size), dtype=np.uint8)
    img[1:-1, 1:-1] = 1
    top, left = (size - hole[0]) // 2, (size - hole[1]) // 2
    img[top : top + hole[0], left : left + hole[1]] = 0
    return img


class TestFillHoles:
    def test_unlimited_fills_every_hole(self):
        result = preprocess_binary(_ring(32, (12, 12)), fill_holes=True)
        assert result[1:-1, 1:-1].all()

    def test_hole_within_diameter_is_filled(self):
        result = preprocess_binary(
            _ring(32, (4, 6)), fill_holes=True, max_hole_diameter=6
        )
        assert result[1:-1, 1:-1].all()

    def test_hole_wider_than_diameter_on_one_axis_is_kept(self):
        result = preprocess_binary(
            _ring(32, (2, 8)), fill_holes=True, max_hole_diameter=6
        )
        assert not result[15:17, 12:20].any()

    def test_extent_uses_spacing(self):
        ring = _ring(32, (4, 4))
        filled = preprocess_binary(
            ring, fill_holes=True, max_hole_diameter=6, spacing=(1.0, 1.0)
        )
        kept = preprocess_binary(
            ring, fill_holes=True, max_hole_diameter=6, spacing=(2.0, 1.0)
        )
        assert filled[14:18, 14:18].all()
        assert not kept[14:18, 14:18].any()


class TestPreprocessingMargin:
    def test_balls_and_hole_diameter_per_axis(self):
        preprocessing = PreprocessingConfig(
            opening_radius=1,
            closing_iterations=2,
            fill_holes=True,
            max_hole_diameter=3.0,
        )
        assert preprocessing_margin(preprocessing, (3.0, 0.5, 0.5), 3) == (8, 13, 13)

    def test_unlimited_holes_add_nothing(self):
        preprocessing = PreprocessingConfig(closing_iterations=1, fill_holes=True)
        assert preprocessing_margin(preprocessing, None, 2) == (2, 2)


class TestClosingBallFootprint:
    """closing_iterations is the radius of a Euclidean ball/disk footprint,
    matching skimage.morphology.closing(mask, ball(r)) away from the array
    border."""

    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("radius", [1, 2, 4])
    def test_matches_skimage_closing(self, ndim, radius):
        morphology = pytest.importorskip("skimage.morphology")
        rng = np.random.default_rng(0)
        shape = (40, 40) if ndim == 2 else (24, 24, 24)
        noisy = ndi.binary_opening(rng.random(shape) > 0.45)
        # keep a background margin of 2*radius: skimage and maskel only
        # differ in how they treat the array border
        margin = (slice(2 * radius, -2 * radius),) * ndim
        inner = np.zeros(shape, dtype=bool)
        inner[margin] = noisy[margin]
        noisy = inner.astype(np.uint8)
        footprint = (morphology.disk if ndim == 2 else morphology.ball)(radius)

        result = preprocess_binary(noisy, closing_iterations=radius)
        expected = morphology.closing(noisy.astype(bool), footprint)

        np.testing.assert_array_equal(result.astype(bool), expected)

    def test_seals_diagonal_gap_the_cross_cannot(self):
        # 45-degree gap, ~4.2px wide: a cross applied 3 times (diamond,
        # reach 3/sqrt(2) along the diagonal) leaves it open, a disk of
        # radius 3 closes it.
        yy, xx = np.mgrid[0:40, 0:40]
        band = np.abs(xx - yy) > 3
        mask = (band & (yy > 5) & (yy < 34) & (xx > 5) & (xx < 34)).astype(np.uint8)
        cross = ndi.generate_binary_structure(2, 1)

        old = ndi.binary_closing(mask, cross, iterations=3)
        new = preprocess_binary(mask, closing_iterations=3)

        assert not old[20, 20]
        assert new[20, 20]

    def test_does_not_erode_at_array_border(self):
        slab = np.zeros((12, 12), dtype=np.uint8)
        slab[:, 3:9] = 1

        result = preprocess_binary(slab, closing_iterations=3)

        np.testing.assert_array_equal(result, slab)


class TestOpeningBallFootprint:
    @pytest.mark.parametrize("ndim", [2, 3])
    @pytest.mark.parametrize("radius", [1, 2, 4])
    def test_matches_skimage_opening(self, ndim, radius):
        morphology = pytest.importorskip("skimage.morphology")
        rng = np.random.default_rng(1)
        shape = (40, 40) if ndim == 2 else (24, 24, 24)
        noisy = ndi.binary_closing(rng.random(shape) > 0.55)
        margin = (slice(2 * radius, -2 * radius),) * ndim
        inner = np.zeros(shape, dtype=bool)
        inner[margin] = noisy[margin]
        footprint = (morphology.disk if ndim == 2 else morphology.ball)(radius)

        result = preprocess_binary(inner.astype(np.uint8), opening_radius=radius)
        expected = morphology.opening(inner, footprint)

        np.testing.assert_array_equal(result.astype(bool), expected)

    def test_cuts_thin_bridge_between_blocks(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[4:16, 2:12] = 1
        mask[4:16, 18:28] = 1
        mask[9:11, 12:18] = 1  # 2px-thick bridge

        result = preprocess_binary(mask, opening_radius=2)

        assert not result[9:11, 13:17].any()
        assert result[6:14, 4:10].all()
        assert result[6:14, 20:26].all()

    def test_does_not_erode_at_array_border(self):
        slab = np.zeros((12, 12), dtype=np.uint8)
        slab[:, 3:9] = 1

        result = preprocess_binary(slab, opening_radius=2)

        np.testing.assert_array_equal(result, slab)

    def test_runs_before_closing(self):
        # A 1px speck 2px away from a block: opening first removes it, so
        # the closing has nothing left to bridge it to the block with.
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:16, 4:12] = 1
        mask[10, 14] = 1

        both = preprocess_binary(mask, opening_radius=1, closing_iterations=2)
        closing_only = preprocess_binary(mask, closing_iterations=2)

        assert closing_only[10, 12:15].all()
        assert not both[10, 13:15].any()


class TestPreprocessBinaryNoOp:
    def test_returns_same_object_when_nothing_requested(self):
        binary = np.zeros((8, 8), dtype=np.uint8)
        binary[2:6, 2:6] = 1

        result = preprocess_binary(binary, closing_iterations=0, fill_holes=False)

        assert result is binary


class TestLabelCompact:
    """`_label_compact` tries a smaller label dtype before falling back to
    scipy's default - see its docstring in maskel/_utils.py for why.
    """

    def test_matches_default_label_dtype(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[2:5, 2:5] = True
        mask[10:14, 10:14] = True
        mask[20:25, 5:9] = True

        labels, num_features = _label_compact(mask)
        expected_labels, expected_num_features = ndi.label(mask)

        assert num_features == expected_num_features
        np.testing.assert_array_equal(labels, expected_labels)

    def test_falls_back_when_compact_dtype_overflows(self):
        # One isolated foreground pixel per cell on a grid -- comfortably
        # more connected components than an 8-bit label array can hold, to
        # force the fallback path without needing scipy's real (16-bit)
        # default to overflow.
        mask = np.zeros((40, 40), dtype=bool)
        mask[::2, ::2] = True
        expected_num_components = int(mask.sum())
        assert expected_num_components > np.iinfo(np.int8).max

        labels, num_features = _label_compact(mask, dtype=np.int8)
        expected_labels, expected_num_features = ndi.label(mask)

        assert num_features == expected_num_components
        assert num_features == expected_num_features
        np.testing.assert_array_equal(labels, expected_labels)
        # the fallback ran scipy's own default, not the overflowing dtype
        assert labels.dtype != np.int8
