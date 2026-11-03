import warnings

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from maskel.config import (
    FeatureConfig,
    OutputConfig,
    PipelineConfig,
    PostprocessingConfig,
    PreprocessingConfig,
)
from maskel.features import analyze
from maskel.io.voxels import read_voxels
from maskel.pipeline import run
from maskel.postprocessing import postprocess
from maskel.skeletonize import skeletonize
from maskel.tiling import TileGrid


def _blobs(shape, seed=0, threshold=0.55):
    rng = np.random.default_rng(seed)
    return (ndi.gaussian_filter(rng.random(shape), 2) > threshold).astype(np.uint8)


def _instances(shape, seed=0):
    labels, _ = ndi.label(_blobs(shape, seed))
    return labels.astype(np.uint16) * 3  # label values need not be consecutive


def _same(a, b):
    for name in ("coords", "label", "labels", "mask_voxels", "crop_voxels"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    np.testing.assert_allclose(a.radius, b.radius)


class TestTileGrid:
    def test_one_tile_without_tile_shape(self):
        grid = TileGrid.build((5, 7))
        (tile,) = grid
        assert tile.core == tile.read == (slice(0, 5), slice(0, 7))

    @pytest.mark.filterwarnings("ignore:halo")
    def test_cores_partition_the_volume_and_halo_is_per_axis(self):
        grid = TileGrid.build((10, 9), tile_shape=4, halo=(1, 2))
        covered = np.zeros((10, 9), int)
        for tile in grid:
            covered[tile.core] += 1
        assert (covered == 1).all()
        tile = list(grid)[4]  # the middle tile
        assert tile.read == (slice(3, 9), slice(2, 9))


class TestSkeletonize:
    def test_binary_mask_is_one_label(self):
        voxels = skeletonize(_blobs((40, 40)) * 255, PreprocessingConfig())
        assert voxels.labels.tolist() == [1]
        assert (voxels.label == 1).all()
        assert voxels.mask_voxels[0] == np.count_nonzero(_blobs((40, 40)))

    def test_instance_labels_are_kept(self):
        mask = _instances((48, 48))
        voxels = skeletonize(mask, PreprocessingConfig())
        assert voxels.labels.tolist() == sorted(set(np.unique(mask)) - {0})

    @pytest.mark.parametrize(
        "preprocessing",
        [
            PreprocessingConfig(),
            PreprocessingConfig(opening_radius=1, closing_iterations=1),
            PreprocessingConfig(fill_holes=True, max_hole_diameter=3.0),
        ],
    )
    @pytest.mark.parametrize("make", [_blobs, _instances])
    def test_tiled_matches_one_tile(self, preprocessing, make):
        mask = make((24, 40, 40), seed=2)
        spacing = (2.0, 1.0, 1.0)
        whole = skeletonize(mask, preprocessing, spacing=spacing)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tiled = skeletonize(
                mask, preprocessing, spacing=spacing, tile_shape=(10, 16, 16), halo=12
            )
        _same(whole, tiled)

    @pytest.mark.filterwarnings("ignore:halo", "ignore:Largest radius")
    def test_parallel_tiles_match_serial(self, tmp_path):
        mask = _instances((24, 40, 40), seed=2)
        with h5py.File(tmp_path / "mask.h5", "w") as f:
            f["mask"] = mask
        preprocessing = PreprocessingConfig(closing_iterations=1)
        kwargs = {"tile_shape": (12, 20, 20), "halo": 8}
        serial = skeletonize(mask, preprocessing, **kwargs)
        with h5py.File(tmp_path / "mask.h5", "r") as f:
            parallel = skeletonize(
                f["mask"],
                preprocessing,
                jobs=2,
                source=(tmp_path / "mask.h5", "mask"),
                **kwargs,
            )
        _same(serial, parallel)

    @pytest.mark.filterwarnings("ignore:halo")
    def test_parallel_tiles_need_a_file(self):
        with pytest.raises(ValueError, match="HDF5"):
            skeletonize(_blobs((40, 40)), PreprocessingConfig(), tile_shape=20, jobs=2)

    @pytest.mark.filterwarnings("ignore:halo")
    def test_unlimited_hole_filling_needs_one_tile(self):
        with pytest.raises(ValueError, match="single tile"):
            skeletonize(
                _blobs((40, 40)), PreprocessingConfig(fill_holes=True), tile_shape=20
            )

    def test_warns_when_halo_is_small_for_the_radii(self):
        mask = np.zeros((40, 40), np.uint8)
        mask[10:30, 2:38] = 1
        with pytest.warns(UserWarning, match="needs a halo"):
            skeletonize(mask, PreprocessingConfig(), tile_shape=20, halo=5)


class TestAnalyze:
    def test_coordinates_are_global(self):
        mask = _blobs((40, 40))
        shifted = np.zeros((60, 70), np.uint8)
        shifted[20:, 30:] = mask
        features = FeatureConfig(branches=True, nodes=True, radius_features=True)

        def tables(m):
            voxels = skeletonize(m, PreprocessingConfig())
            (result,) = analyze(postprocess(voxels, PostprocessingConfig()), features)
            return pd.DataFrame(result.branch_records), pd.DataFrame(
                result.node_records
            )

        branches, nodes = tables(mask)
        branches2, nodes2 = tables(shifted)
        np.testing.assert_array_equal(
            branches2["image-coord-src-1"], branches["image-coord-src-1"] + 30
        )
        np.testing.assert_array_equal(nodes2["coord_0"], nodes["coord_0"] + 20)
        np.testing.assert_allclose(
            branches2["branch-distance"], branches["branch-distance"]
        )
        np.testing.assert_allclose(nodes2["radius"], nodes["radius"])

    def test_one_result_per_label_with_its_id(self):
        mask = _instances((48, 48))
        voxels = skeletonize(mask, PreprocessingConfig())
        results = list(analyze(voxels, FeatureConfig()))
        assert {r.summary["object_id"] for r in results} <= set(np.unique(mask))
        assert len(results) > 1


class TestRun:
    @pytest.mark.filterwarnings("ignore:halo")
    def test_writes_voxel_lists_and_tables(self, tmp_path):
        with h5py.File(tmp_path / "vessels.h5", "w") as f:
            f["mask"] = _blobs((40, 40))
        config = PipelineConfig(
            postprocessing=PostprocessingConfig(junction_cleanup=True),
            feature_extraction=FeatureConfig(branches=True),
            output=OutputConfig(write_branch_csv=True, write_graphml=True),
        )
        run(tmp_path / "vessels.h5", tmp_path / "out", config, tile_shape=20, halo=30)

        out = tmp_path / "out"
        assert read_voxels(out / "vessels_voxels.h5").attrs["stage"] == "skeletonize"
        assert (
            read_voxels(out / "vessels_voxels_post.h5").attrs["stage"] == "postprocess"
        )
        assert (out / "vessels_summary.csv").is_file()
        assert (out / "vessels_branches.csv").is_file()
        assert (out / "vessels_1_graph.graphml").is_file()
        assert not (out / "vessels_voxels_mask.h5").exists()

    def test_writes_skeleton_masks(self, tmp_path):
        with h5py.File(tmp_path / "vessels.h5", "w") as f:
            f["mask"] = _blobs((40, 40))
        config = PipelineConfig(output=OutputConfig(write_skeleton_mask=True))
        run(tmp_path / "vessels.h5", tmp_path / "out", config)

        out = tmp_path / "out"
        for stem in ("vessels_voxels", "vessels_voxels_post"):
            voxels = read_voxels(out / f"{stem}.h5")
            with h5py.File(out / f"{stem}_mask.h5", "r") as f:
                mask = f["skeleton"][()]
            assert mask.shape == voxels.shape
            np.testing.assert_array_equal(
                np.argwhere(mask), np.unique(voxels.coords, axis=0)
            )
