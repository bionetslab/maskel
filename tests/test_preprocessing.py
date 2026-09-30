import numpy as np

from maskel.config import ExtractionConfig, OutputConfig, PipelineConfig
from maskel.pipeline import analyze_segmentation_mask


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
