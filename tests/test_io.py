import h5py
import numpy as np
import pytest
import tifffile
from PIL import Image

from maskel.io.images import load_image, open_image
from maskel.io.tables import write_csv
from maskel.io.voxels import VoxelList, read_voxels, write_voxels


class TestLoadImage:
    def test_load_png(self, tmp_path):
        path = tmp_path / "test.png"
        Image.fromarray(np.zeros((3, 3), dtype=np.uint8)).save(path)
        arr = load_image(path)
        assert arr.shape == (3, 3)

    def test_load_jpg(self, tmp_path):
        path = tmp_path / "test.jpg"
        Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 128).save(path)
        arr = load_image(path)
        assert arr.shape == (4, 4)

    def test_load_npy(self, tmp_path):
        path = tmp_path / "test.npy"
        np.save(path, np.ones((5, 5), dtype=np.uint8))
        arr = load_image(path)
        assert arr.shape == (5, 5)

    def test_load_rgb_collapses_to_grayscale(self, tmp_path):
        path = tmp_path / "rgb.png"
        rgb = np.zeros((8, 8, 3), dtype=np.uint8)
        rgb[2:6, 2:6, 0] = 255
        Image.fromarray(rgb).save(path)
        arr = load_image(path)
        assert arr.ndim == 2
        assert arr.shape == (8, 8)
        assert arr[4, 4] == 255
        assert arr[0, 0] == 0

    def test_load_rgba_collapses_to_grayscale(self, tmp_path):
        path = tmp_path / "rgba.png"
        rgba = np.zeros((6, 6, 4), dtype=np.uint8)
        rgba[:, :, :3] = 100
        Image.fromarray(rgba).save(path)
        arr = load_image(path)
        assert arr.ndim == 2

    def test_load_2d_passes_through(self, tmp_path):
        path = tmp_path / "gray.png"
        Image.fromarray(np.eye(10, dtype=np.uint8) * 255).save(path)
        arr = load_image(path)
        assert arr.ndim == 2

    def test_load_3d_npy(self, tmp_path):
        path = tmp_path / "vol.npy"
        np.save(path, np.ones((3, 4, 5), dtype=np.uint8))
        arr = load_image(path)
        assert arr.shape == (3, 4, 5)

    def test_load_multipage_tiff_stacks_to_3d(self, tmp_path):
        path = tmp_path / "vol.tif"
        vol = np.zeros((6, 4, 5), dtype=np.uint8)
        for z in range(6):
            vol[z, z % 4, z % 5] = 255
        frames = [Image.fromarray(sl) for sl in vol]
        frames[0].save(path, save_all=True, append_images=frames[1:])
        arr = load_image(path)
        assert arr.shape == (6, 4, 5)
        assert np.array_equal(arr, vol)

    def test_load_single_page_tiff_stays_2d(self, tmp_path):
        path = tmp_path / "flat.tif"
        Image.fromarray(np.eye(7, dtype=np.uint8) * 255).save(path)
        arr = load_image(path)
        assert arr.shape == (7, 7)

    def test_load_single_page_rgb_tiff_collapses_channels(self, tmp_path):
        path = tmp_path / "rgb.tif"
        rgb = np.zeros((4, 5, 3), dtype=np.uint8)
        rgb[1, 2, 0] = 200
        Image.fromarray(rgb, mode="RGB").save(path)
        arr = load_image(path)
        assert arr.shape == (4, 5)
        assert arr[1, 2] == 200

    def test_load_imagej_stack_with_singleton_channel(self, tmp_path):
        """tifffile drops a 1-length channel axis, so ZCYX arrives as 3D."""
        path = tmp_path / "hyper.tif"
        vol = np.zeros((4, 1, 8, 9), dtype=np.uint8)
        vol[2, 0, 3, 4] = 255
        tifffile.imwrite(path, vol, imagej=True, metadata={"axes": "ZCYX"})
        arr = load_image(path)
        assert arr.shape == (4, 8, 9)
        assert arr[2, 3, 4] == 255

    def test_load_multichannel_stack_raises(self, tmp_path):
        path = tmp_path / "multichan.tif"
        tifffile.imwrite(
            path,
            np.zeros((4, 2, 8, 9), dtype=np.uint8),
            imagej=True,
            metadata={"axes": "ZCYX"},
        )
        with pytest.raises(
            ValueError, match=r"Expected 2D or 3D image, got shape=\(4, 2, 8, 9\)"
        ):
            load_image(path)

    def test_load_multipage_rgb_tiff_raises(self, tmp_path):
        """A colour stack stays 4D: channels are only collapsed for 2D pages."""
        path = tmp_path / "rgb_vol.tif"
        vol = np.zeros((3, 4, 5, 3), dtype=np.uint8)
        frames = [Image.fromarray(sl, mode="RGB") for sl in vol]
        frames[0].save(path, save_all=True, append_images=frames[1:])
        with pytest.raises(ValueError, match=r"Expected 2D or 3D image"):
            load_image(path)

    def test_load_compressed_tiff_stack(self, tmp_path):
        """Compressed multi-page stack; uses a stdlib codec available on 3.13."""
        path = tmp_path / "lzma_vol.tif"
        vol = (np.arange(3 * 4 * 5, dtype=np.uint8) % 2 * 255).reshape(3, 4, 5)
        tifffile.imwrite(path, vol, compression="lzma", photometric="minisblack")
        arr = load_image(path)
        assert arr.shape == (3, 4, 5)
        assert np.array_equal(arr, vol)

    def test_load_tiff_does_not_use_pillow(self, tmp_path, monkeypatch):
        """Guard the regression: Pillow's reader hands back page 0 only."""
        path = tmp_path / "vol.tif"
        tifffile.imwrite(path, np.zeros((7, 4, 5), dtype=np.uint8))
        monkeypatch.setattr(
            "maskel.io.images.Image.open",
            lambda *a, **k: pytest.fail("TIFF must not be read through Pillow"),
        )
        assert load_image(path).shape == (7, 4, 5)


class TestOpenImage:
    def _h5(self, path, **datasets):
        with h5py.File(path, "w") as f:
            for name, data in datasets.items():
                f.create_dataset(name, data=data, chunks=True, compression="gzip")
        return path

    def test_hdf5_is_read_lazily(self, tmp_path):
        path = self._h5(tmp_path / "a.h5", mask=np.ones((6, 7), np.uint8))
        with open_image(path) as image:
            assert isinstance(image, h5py.Dataset)
            assert image[1:3, 2:5].shape == (2, 3)

    def test_named_dataset_among_several(self, tmp_path):
        path = self._h5(tmp_path / "a.h5", a=np.zeros((4, 4)), b=np.ones((5, 5)))
        with open_image(path, "b") as image:
            assert image.shape == (5, 5)

    def test_ambiguous_datasets_need_a_name(self, tmp_path):
        path = self._h5(tmp_path / "a.h5", a=np.zeros((4, 4)), b=np.ones((5, 5)))
        with pytest.raises(ValueError, match="exactly one"), open_image(path):
            pass

    def test_other_formats_are_loaded_whole(self, tmp_path):
        np.save(tmp_path / "a.npy", np.ones((3, 4), np.uint8))
        with open_image(tmp_path / "a.npy") as image:
            assert isinstance(image, np.ndarray)


class TestVoxels:
    def _voxels(self):
        return VoxelList(
            shape=(10, 10),
            spacing=(2.0, 0.5),
            coords=np.array([[5, 5], [1, 2], [1, 1]]),
            radius=np.array([3.0, 2.0, 1.0]),
            label=np.array([4, 7, 4]),
            labels=np.array([4, 7]),
            mask_voxels=np.array([20, 30]),
            crop_voxels=np.array([50, 60]),
            attrs={"stage": "skeletonize"},
        )

    def test_sorted_by_label_then_position(self):
        voxels = self._voxels()
        np.testing.assert_array_equal(voxels.coords, [[1, 1], [5, 5], [1, 2]])
        np.testing.assert_array_equal(voxels.radius, [1.0, 3.0, 2.0])

    def test_iterates_per_label(self):
        parts = list(self._voxels())
        assert [p.label for p in parts] == [4, 7]
        assert [len(p.coords) for p in parts] == [2, 1]
        assert parts[1].mask_voxels == 30

    def test_round_trip(self, tmp_path):
        voxels = self._voxels()
        write_voxels(tmp_path / "v.h5", voxels)
        loaded = read_voxels(tmp_path / "v.h5")
        np.testing.assert_array_equal(loaded.coords, voxels.coords)
        np.testing.assert_array_equal(loaded.radius, voxels.radius)
        np.testing.assert_array_equal(loaded.crop_voxels, voxels.crop_voxels)
        assert loaded.shape == (10, 10)
        assert loaded.spacing == (2.0, 0.5)
        assert loaded.attrs == {"stage": "skeletonize"}


class TestWriteCsv:
    def test_union_of_keys_as_sorted_columns(self, tmp_path):
        write_csv(tmp_path / "a.csv", [{"b": np.int64(1)}, {"a": 2.5}])
        assert (tmp_path / "a.csv").read_text().splitlines() == ["a,b", ",1", "2.5,"]

    def test_empty_rows_write_nothing(self, tmp_path):
        write_csv(tmp_path / "a.csv", [])
        assert not (tmp_path / "a.csv").exists()
