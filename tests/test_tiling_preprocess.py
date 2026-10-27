"""Tiled preprocessing must give exactly the whole-volume result.

Each block is preprocessed on its own with edge-repeated padding at the block
edge, so voxels near that edge come out wrong. `BlockPreprocess.margin` has to
be wide enough that the error never reaches a tile's core.
"""

import numpy as np
import pytest
from scipy import ndimage as ndi

from maskel.config import ExtractionConfig
from maskel.preprocessing import preprocess_binary
from maskel.thin import lee94_thin
from maskel.tiling import BlockPreprocess, TileGrid, preprocess_from_config
from maskel.tiling.stream import skeletonize_streaming

pytestmark = pytest.mark.filterwarnings("ignore:halo=.*reads")


def _blobs(shape, seed, sigma=1.5, level=0.55):
    """Smooth random structures with bumps, pits, thin bridges and holes."""
    rng = np.random.default_rng(seed)
    noise = ndi.gaussian_filter(rng.random(shape), sigma)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return (noise > level).astype(np.uint8)


def _tiled(mask, preprocess, tile, extra_halo=0):
    """Preprocess *mask* block by block and stitch the cores back together."""
    out = np.zeros_like(mask)
    grid = TileGrid(mask.shape, tile, halo=preprocess.margin + extra_halo)
    for spec in grid:
        block = preprocess.apply(mask[spec.read].copy())
        out[spec.core] = block[spec.core_in_read]
    return out


CONFIGS = {
    "opening": BlockPreprocess(opening_radius=2),
    "closing": BlockPreprocess(closing_iterations=2),
    "fill_holes": BlockPreprocess(fill_holes=True, max_hole_size=6),
    "all": BlockPreprocess(
        opening_radius=1,
        closing_iterations=2,
        fill_holes=True,
        max_hole_size=6,
    ),
}


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("shape", [(70, 90), (30, 34, 38)])
def test_tiled_preprocessing_matches_whole_volume(name, shape):
    preprocess = CONFIGS[name]
    mask = _blobs(shape, seed=len(shape))

    expected = preprocess.apply(mask)
    # Tiles much smaller than the margin put a seam next to almost every voxel.
    actual = _tiled(mask, preprocess, tile=7)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("name", ["opening", "closing"])
def test_one_voxel_less_margin_breaks_it(name):
    """The 2r per opening/closing is tight - the equality above isn't luck."""
    preprocess = CONFIGS[name]
    mask = _blobs((70, 90), seed=2)

    expected = preprocess.apply(mask)
    actual = _tiled(mask, preprocess, tile=7, extra_halo=-1)

    assert not np.array_equal(actual, expected)


def test_margin_adds_up_every_step():
    p = BlockPreprocess(
        opening_radius=2,
        closing_iterations=3,
        fill_holes=True,
        max_hole_size=10,
    )
    assert p.margin == 2 * 2 + 2 * 3 + 10


def test_preprocess_from_config_maps_new_steps():
    extraction = ExtractionConfig(opening_radius=2, closing_iterations=1)
    p = preprocess_from_config(extraction)
    assert (p.opening_radius, p.closing_iterations) == (2, 1)

    assert not preprocess_from_config(ExtractionConfig())


def test_negative_opening_radius_is_rejected():
    with pytest.raises(ValueError, match="opening_radius"):
        BlockPreprocess(opening_radius=-1)


def test_manifest_record_includes_new_steps():
    record = BlockPreprocess(opening_radius=2).to_dict()
    assert record["opening_radius"] == 2


def test_streaming_skeleton_matches_untiled(tmp_path):
    """End to end: tiled preprocessing + thinning == whole-volume result."""
    # Tubes of radius ~2 crossing the volume, with jagged surfaces.
    shape = (36, 40, 44)
    zz, yy, xx = np.indices(shape)
    tubes = ((yy - 20) ** 2 + (zz - 18) ** 2 <= 5) | (
        (xx - 22) ** 2 + (zz - 18) ** 2 <= 5
    )
    rng = np.random.default_rng(0)
    mask = (tubes ^ (rng.random(shape) < 0.03) & tubes).astype(np.uint8)

    preprocess = BlockPreprocess(opening_radius=1, closing_iterations=1)
    out = skeletonize_streaming(
        mask, tmp_path, tile_shape=16, halo=12, preprocess=preprocess
    )

    expected = lee94_thin(
        preprocess_binary(mask, opening_radius=1, closing_iterations=1)
    )
    np.testing.assert_array_equal(np.load(out), expected)
