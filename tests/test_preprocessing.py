import numpy as np
import pytest
from scipy import ndimage as ndi

from maskel.config import ExtractionConfig, OutputConfig, PipelineConfig
from maskel.pipeline import _label_compact, analyze_segmentation_mask, preprocess_binary


class TestClosing:
    def test_closing_bridges_1px_gap_in_thick_line(self):
        gap = np.zeros((16, 16), dtype=np.uint8)
        gap[7:10, 4:7] = 1
        gap[7:10, 8:12] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(closing_iterations=1),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(gap, config)
        # closing bridges the 1px gap at column 7 in the 3px-thick line
        assert result.skeleton[7:10, 7].any()

    def test_closing_bridges_gap_in_thick_region(self):
        block = np.zeros((16, 16), dtype=np.uint8)
        block[6:10, 4:7] = 1
        block[6:10, 8:12] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(closing_iterations=1),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(block, config)
        # the 1px gap at column 7 should be bridged before thinning
        assert result.skeleton[6:10, 7].any()

    def test_closing_off_does_not_modify(self):
        gap = np.zeros((16, 16), dtype=np.uint8)
        gap[7:10, 4:7] = 1
        gap[7:10, 8:12] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(closing_iterations=0),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(gap, config)
        # without closing the gap remains
        assert not result.skeleton[7:10, 7].any()


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

    def test_pipeline_enables_preprocessing_for_opening_alone(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[4:16, 2:12] = 1
        mask[4:16, 18:28] = 1
        mask[9:11, 12:18] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(opening_radius=2), output=OutputConfig()
        )
        result = analyze_segmentation_mask(mask, config)

        assert result.preprocessed_binary is not None
        assert not result.preprocessed_binary[9:11, 13:17].any()


class TestFillHoles:
    def test_fill_holes_fills_enclosed_void(self):
        # The 1px background border is what keeps a 0 in the mask once the hole
        # is filled - an all-foreground object is not valid thinning input.
        ring = np.zeros((16, 16), dtype=np.uint8)
        ring[1:15, 1:15] = 1
        ring[6:10, 6:10] = 0
        ring[6:10, 6] = 1
        ring[6:10, 9] = 1
        ring[6, 6:10] = 1
        ring[9, 6:10] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(fill_holes=True),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(ring, config)
        # preprocessed_binary should have the hole filled
        assert result.preprocessed_binary is not None
        assert result.preprocessed_binary[7:9, 7:9].all()


class TestMaxHoleSize:
    def test_max_hole_size_skips_large_holes(self):
        ring = np.ones((32, 32), dtype=np.uint8)
        ring[8:24, 8:24] = 0
        ring[8:24, 8] = 1
        ring[8:24, 23] = 1
        ring[8, 8:24] = 1
        ring[23, 8:24] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(fill_holes=True, max_hole_size=50),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(ring, config)
        # 15x15 = 225 pixels, above threshold, so preprocessed binary keeps it as a hole
        assert result.preprocessed_binary is not None
        assert not result.preprocessed_binary[9:23, 9:23].any()

    def test_max_hole_size_fills_small_holes(self):
        # Background border: see test_fill_holes_fills_enclosed_void.
        ring = np.zeros((16, 16), dtype=np.uint8)
        ring[1:15, 1:15] = 1
        ring[7:9, 7:9] = 0
        ring[7:9, 7] = 1
        ring[7:9, 8] = 1
        ring[7, 7:9] = 1
        ring[8, 7:9] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(fill_holes=True, max_hole_size=50),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(ring, config)
        # 2x2 = 4 pixels, under threshold, so filled in preprocessed
        assert result.preprocessed_binary is not None
        assert result.preprocessed_binary[7:9, 7:9].all()

    def test_max_hole_size_zero_fills_all(self):
        # Background border: see test_fill_holes_fills_enclosed_void.
        ring = np.zeros((16, 16), dtype=np.uint8)
        ring[1:15, 1:15] = 1
        ring[7:11, 7:11] = 0
        ring[7:11, 7] = 1
        ring[7:11, 10] = 1
        ring[7, 7:11] = 1
        ring[10, 7:11] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(fill_holes=True, max_hole_size=0),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(ring, config)
        # zero = unlimited, so 4x4 hole gets filled
        assert result.preprocessed_binary is not None
        assert result.preprocessed_binary[8:10, 8:10].all()


class TestPreprocessedBinary:
    def test_set_when_closing_active(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        img[8, 4:12] = 1
        config = PipelineConfig(
            extraction=ExtractionConfig(closing_iterations=1),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(img, config)
        assert result.preprocessed_binary is not None
        assert result.preprocessed_binary.shape == img.shape

    def test_set_when_fill_holes_active(self):
        # Background border: see TestFillHoles.
        ring = np.zeros((16, 16), dtype=np.uint8)
        ring[1:15, 1:15] = 1
        ring[7:9, 7:9] = 0
        ring[7:9, 7] = 1
        ring[7:9, 8] = 1
        ring[7, 7:9] = 1
        ring[8, 7:9] = 1
        config = PipelineConfig(
            extraction=ExtractionConfig(fill_holes=True),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(ring, config)
        assert result.preprocessed_binary is not None

    def test_none_when_preprocessing_disabled(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        img[8, 4:12] = 1
        config = PipelineConfig(
            extraction=ExtractionConfig(),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(img, config)
        assert result.preprocessed_binary is None

    def test_none_when_show_preprocessed_checked_but_no_preprocessing(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        img[8, 4:12] = 1
        config = PipelineConfig(
            extraction=ExtractionConfig(show_preprocessed=True),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(img, config)
        assert result.preprocessed_binary is None


class TestPreprocessedBinaryMultiObject:
    """Despite its name, preprocessed_binary carries real object ids for a
    multi-object instance segmentation map, not just 0/1 - see
    analyze_segmentation_mask's stitching of full_preprocessed."""

    def test_preserves_object_ids(self):
        mask = np.zeros((16, 32), dtype=np.uint8)
        mask[4:8, 4:8] = 5
        mask[4:8, 20:24] = 9

        config = PipelineConfig(
            extraction=ExtractionConfig(closing_iterations=1),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(mask, config)

        assert result.preprocessed_binary is not None
        assert (result.preprocessed_binary[4:8, 4:8] == 5).all()
        assert (result.preprocessed_binary[4:8, 20:24] == 9).all()
        assert (result.preprocessed_binary[0:4, 0:16] == 0).all()

    def test_plain_binary_mask_still_reports_id_one(self):
        img = np.zeros((16, 16), dtype=np.uint8)
        img[8, 4:12] = 1

        config = PipelineConfig(
            extraction=ExtractionConfig(closing_iterations=1),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(img, config)

        assert result.preprocessed_binary is not None
        foreground = result.preprocessed_binary[result.preprocessed_binary != 0]
        assert (foreground == 1).all()

    def test_dtype_matches_integer_input_mask(self):
        mask = np.zeros((16, 16), dtype=np.int32)
        mask[4:8, 4:8] = 5

        config = PipelineConfig(
            extraction=ExtractionConfig(closing_iterations=1),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(mask, config)

        assert result.preprocessed_binary.dtype == mask.dtype

    def test_dtype_is_integer_for_non_integer_input_mask(self):
        mask = np.zeros((16, 16), dtype=np.float64)
        mask[4:8, 4:8] = 1.0

        config = PipelineConfig(
            extraction=ExtractionConfig(closing_iterations=1),
            output=OutputConfig(),
        )
        result = analyze_segmentation_mask(mask, config)

        assert np.issubdtype(result.preprocessed_binary.dtype, np.integer)


class TestGaussianFilter:
    def test_fills_pit_and_removes_bump(self):
        block = np.zeros((20, 20), dtype=np.uint8)
        block[5:15, 5:15] = 1
        block[5, 9] = 0  # 1px pit in the top edge
        block[15, 9] = 1  # 1px bump on the bottom edge

        result = preprocess_binary(block, gaussian_sigma=1.0, gaussian_threshold=0.5)

        assert result.dtype == block.dtype
        assert result[5, 9] == 1
        assert result[15, 9] == 0
        # the body of the block is untouched
        assert result[6:14, 6:14].all()

    def test_does_not_erode_at_array_border(self):
        # an object running into the array edge must not lose voxels there
        slab = np.zeros((12, 12), dtype=np.uint8)
        slab[:, 3:9] = 1

        result = preprocess_binary(slab, gaussian_sigma=1.0)

        np.testing.assert_array_equal(result, slab)

    def test_threshold_controls_growth(self):
        block = np.zeros((20, 20), dtype=np.uint8)
        block[5:15, 5:15] = 1

        low = preprocess_binary(block, gaussian_sigma=1.0, gaussian_threshold=0.2)
        high = preprocess_binary(block, gaussian_sigma=1.0, gaussian_threshold=0.8)

        assert low.sum() > block.sum() > high.sum()

    def test_runs_before_closing(self):
        # Two 3px-thick bars with a 3px gap: closing(1) alone can't bridge it
        # (see TestClosing), but blurring with a low threshold first grows
        # both bars enough that the closing then fills what's left.
        bars = np.zeros((16, 20), dtype=np.uint8)
        bars[6:9, 2:8] = 1
        bars[6:9, 11:18] = 1

        closing_only = preprocess_binary(bars, closing_iterations=1)
        both = preprocess_binary(
            bars, closing_iterations=1, gaussian_sigma=1.0, gaussian_threshold=0.2
        )

        assert not closing_only[7, 8:11].all()
        assert both[7, 8:11].all()

    def test_pipeline_applies_it_only_when_enabled(self):
        block = np.zeros((20, 20), dtype=np.uint8)
        block[5:15, 5:15] = 1
        block[5, 9] = 0

        def run(enabled: bool):
            config = PipelineConfig(
                extraction=ExtractionConfig(
                    gaussian_filter=enabled, gaussian_sigma=1.0, show_preprocessed=True
                ),
                output=OutputConfig(),
            )
            return analyze_segmentation_mask(block, config)

        assert run(False).preprocessed_binary is None
        preprocessed = run(True).preprocessed_binary
        assert preprocessed is not None
        assert preprocessed[5, 9] == 1


class TestPreprocessBinaryNoOp:
    def test_returns_same_object_when_nothing_requested(self):
        binary = np.zeros((8, 8), dtype=np.uint8)
        binary[2:6, 2:6] = 1

        result = preprocess_binary(binary, closing_iterations=0, fill_holes=False)

        assert result is binary


class TestLabelCompact:
    """`_label_compact` tries a smaller label dtype before falling back to
    scipy's default - see its docstring in maskel/pipeline.py for why.
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
