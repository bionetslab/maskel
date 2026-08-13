import io
import json
from math import prod

import numpy as np
import pytest

from vesskel._utils import to_binary
from vesskel.config import ExtractionConfig, OutputConfig, PipelineConfig
from vesskel.pipeline import analyze_binary_image
from vesskel.thin import lee94_thin
from vesskel.tiling import (
    DONE_LOG_NAME,
    MANIFEST_NAME,
    PROGRESS_ENV_VAR,
    BlockReader,
    DoneLog,
    TileGrid,
    analyze_tiled,
    read_manifest,
    skeletonize_streaming,
    write_manifest,
)
from vesskel.tiling.progress import bar_glyphs, format_duration, resolve_progress_style


def branching_image(shape: tuple[int, int] = (96, 128), thickness: int = 3):
    """2D grid of thick lines: several junctions, spans many tiles."""
    img = np.zeros(shape, dtype=np.uint8)
    h, w = shape
    img[h // 2 - thickness : h // 2 + thickness, :] = 1
    for col in range(w // 8, w, w // 4):
        img[:, col - 1 : col + 2] = 1
    return img


# `branching_volume` is 4 voxels thick, and 3D thinning sweeps six directions
# per iteration, so its domain of dependence reaches much further than 2D's.
# Measured convergence for these fixtures: 3D needs 16, 2D needs 2.
HALO_3D = 16


def branching_volume(shape: tuple[int, int, int] = (32, 40, 48), thickness: int = 2):
    """3D trunk with perpendicular branches."""
    vol = np.zeros(shape, dtype=np.uint8)
    d, h, w = shape
    vol[
        d // 2 - thickness : d // 2 + thickness,
        h // 2 - thickness : h // 2 + thickness,
        :,
    ] = 1
    for x in range(w // 6, w, w // 3):
        vol[d // 2 - 1 : d // 2 + 2, :, x - 1 : x + 2] = 1
    return vol


def plain_config(**extraction) -> PipelineConfig:
    return PipelineConfig(
        extraction=ExtractionConfig(**extraction), output=OutputConfig()
    )


def tiled_skeleton(image, work_dir, **kwargs) -> np.ndarray:
    """Run the tiled path and read back the skeleton it wrote."""
    return np.load(skeletonize_streaming(image, work_dir, **kwargs))


def assert_partitions(grid: TileGrid):
    """Every voxel is claimed by exactly one core."""
    coverage = np.zeros(grid.shape, dtype=np.int32)
    for spec in grid:
        coverage[spec.core] += 1
    assert np.all(coverage == 1)


class TestTileGridGeometry:
    @pytest.mark.parametrize(
        ("shape", "tile_shape", "halo"),
        [
            ((64, 64), 16, 4),
            ((64, 64), 16, 0),
            ((70, 53), 16, 5),  # not a multiple: short last tile
            ((70, 53), (16, 32), 5),  # anisotropic tiles
            ((30, 40, 50), 16, 3),
            ((9,), 4, 2),  # 1D
        ],
    )
    def test_cores_partition_volume(self, shape, tile_shape, halo):
        assert_partitions(TileGrid(shape=shape, tile_shape=tile_shape, halo=halo))

    def test_read_contains_core_and_stays_in_bounds(self):
        grid = TileGrid(shape=(70, 53), tile_shape=16, halo=5)
        for spec in grid:
            for axis, (core, read) in enumerate(zip(spec.core, spec.read)):
                assert read.start <= core.start and read.stop >= core.stop
                assert read.start >= 0
                assert read.stop <= grid.shape[axis]

    def test_core_in_read_locates_the_core(self):
        """Slicing an array by `read` then `core_in_read` == slicing by `core`."""
        rng = np.random.default_rng(0)
        volume = rng.integers(0, 255, size=(37, 41), dtype=np.uint8)
        grid = TileGrid(shape=volume.shape, tile_shape=8, halo=3)
        for spec in grid:
            block = volume[spec.read]
            assert np.array_equal(block[spec.core_in_read], volume[spec.core])

    def test_interior_tile_has_full_halo(self):
        grid = TileGrid(shape=(64, 64), tile_shape=16, halo=4)
        interior = next(s for s in grid if s.index == (1, 1))
        assert interior.read_shape == (24, 24)
        assert interior.core_shape == (16, 16)
        assert interior.core_in_read == (slice(4, 20), slice(4, 20))

    def test_border_tile_halo_is_clipped(self):
        grid = TileGrid(shape=(64, 64), tile_shape=16, halo=4)
        corner = next(s for s in grid if s.index == (0, 0))
        assert corner.read_shape == (20, 20)
        assert corner.core_in_read == (slice(0, 16), slice(0, 16))

    def test_counts_and_len(self):
        grid = TileGrid(shape=(70, 53), tile_shape=16, halo=0)
        assert grid.counts == (5, 4)
        assert len(grid) == 20
        assert len(list(grid)) == 20

    def test_single_tile_when_tile_covers_volume(self):
        grid = TileGrid(shape=(32, 32), tile_shape=64, halo=8)
        specs = list(grid)
        assert len(specs) == 1
        assert specs[0].read_shape == (32, 32)
        assert specs[0].core_in_read == (slice(0, 32), slice(0, 32))

    def test_scalar_tile_shape_is_broadcast(self):
        assert TileGrid(shape=(10, 20, 30), tile_shape=8, halo=0).tile_shape == (
            8,
            8,
            8,
        )

    def test_read_overhead(self):
        # Per axis: 10 tiles, the two edge ones clipped to 110 and the eight
        # interior ones 120 -> 1180 read per 1000, squared over two axes.
        grid = TileGrid(shape=(1000, 1000), tile_shape=100, halo=10)
        assert grid.read_overhead == pytest.approx(1.18**2)

    @pytest.mark.parametrize(
        ("shape", "tile_shape", "halo"),
        [
            ((1000, 1000), 100, 10),
            ((548, 4526, 496), 256, 100),
            ((70, 53), (16, 32), 5),
            ((64, 64), 16, 0),
        ],
    )
    def test_read_overhead_matches_the_tiles_actually_read(
        self, shape, tile_shape, halo
    ):
        grid = TileGrid(shape=shape, tile_shape=tile_shape, halo=halo)
        actual = sum(prod(s.read_shape) for s in grid)
        assert grid.read_total == actual
        assert grid.read_overhead == pytest.approx(actual / prod(grid.shape))

    @pytest.mark.parametrize(
        ("shape", "tile_shape", "halo"),
        [
            ((548, 4526, 4969), 1024, 100),  # tile exceeds the first axis
            ((100, 100), 512, 50),  # tile exceeds every axis
            ((64, 64), 16, 0),  # no halo at all
            ((70, 53), (16, 32), 5),
        ],
    )
    def test_read_overhead_is_never_below_one(self, shape, tile_shape, halo):
        """The cores cover the volume once; a halo can only add to that."""
        assert (
            TileGrid(shape=shape, tile_shape=tile_shape, halo=halo).read_overhead >= 1
        )

    def test_axis_shorter_than_the_tile_adds_no_overhead(self):
        """That axis is never split, so it is read exactly once."""
        grid = TileGrid(shape=(548, 4526, 4969), tile_shape=1024, halo=100)
        flat = TileGrid(shape=(4526, 4969), tile_shape=1024, halo=100)
        assert grid.read_overhead == pytest.approx(flat.read_overhead)

    def test_warns_on_excessive_halo(self):
        with pytest.warns(UserWarning, match="the volume"):
            TileGrid(shape=(1000, 1000), tile_shape=32, halo=32)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"shape": (10, 10), "tile_shape": (4,), "halo": 1}, "axes"),
            ({"shape": (10, 10), "tile_shape": 0, "halo": 1}, "positive"),
            ({"shape": (10, 0), "tile_shape": 4, "halo": 1}, "positive"),
            ({"shape": (10, 10), "tile_shape": 4, "halo": -1}, "non-negative"),
            ({"shape": (), "tile_shape": 4, "halo": 1}, "at least one axis"),
        ],
    )
    def test_invalid_grid_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            TileGrid(**kwargs)


class TestProgress:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.4, "<1s"),
            (1, "1s"),
            (45, "45s"),
            (60, "1m00s"),
            (192, "3m12s"),
            (3600, "1h00m"),
            (7620, "2h07m"),
        ],
    )
    def test_format_duration(self, seconds, expected):
        assert format_duration(seconds) == expected

    def test_bar_rewrites_a_single_line(self, tmp_path, capsys):
        grid = TileGrid(shape=(48, 48), tile_shape=16, halo=4)
        skeletonize_streaming(
            branching_image((48, 48)), tmp_path, tile_shape=16, halo=4, progress="bar"
        )

        bar, tail = capsys.readouterr().out.split("\n", 1)
        # One line for all 9 tiles: 10 paints (initial + one per tile), each
        # prefixed by a carriage return rather than a newline.
        assert bar.count("\r") == len(grid) + 1
        assert "100.0%" in bar
        assert "9/9 tiles" in bar
        assert "ETA" in bar
        assert tail.strip().startswith("Thinning done in")

    def test_bar_repaint_never_shrinks(self, tmp_path, capsys):
        """A shorter repaint would leave a tail of the previous line behind."""
        skeletonize_streaming(
            branching_image((48, 48)), tmp_path, tile_shape=16, halo=4, progress="bar"
        )

        paints = capsys.readouterr().out.split("\n")[0].split("\r")[1:]
        widths = [len(p) for p in paints]
        assert widths == sorted(widths)

    def test_lines_style_is_bounded_and_newline_terminated(self, tmp_path, capsys):
        grid = TileGrid(shape=(120, 16), tile_shape=(4, 16), halo=0)
        assert len(grid) == 30
        skeletonize_streaming(
            np.zeros(grid.shape, dtype=np.uint8),
            tmp_path,
            tile_shape=(4, 16),
            halo=0,
            progress="lines",
        )

        out = capsys.readouterr().out
        assert "\r" not in out
        # ~one line per 5% plus the final one, not one per tile.
        assert len(out.splitlines()) < len(grid)
        assert "100.0%" in out
        assert "Thinning done in" in out

    def test_thinning_is_labelled(self, tmp_path, capsys):
        skeletonize_streaming(
            branching_image((48, 48)), tmp_path, tile_shape=16, halo=4, progress="bar"
        )

        out = capsys.readouterr().out
        assert out.startswith("\rThinning [")
        assert "9/9 tiles" in out
        assert "Thinning done in" in out

    def test_quiet_by_default(self, tmp_path, capsys):
        skeletonize_streaming(
            branching_image((48, 48)), tmp_path, tile_shape=16, halo=4
        )
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(("given", "expected"), [(True, "lines"), (False, "off")])
    def test_bools_still_accepted(self, given, expected, monkeypatch):
        monkeypatch.delenv("PYCHARM_HOSTED", raising=False)
        monkeypatch.delenv(PROGRESS_ENV_VAR, raising=False)
        assert resolve_progress_style(given, io.StringIO()) == expected

    def test_auto_picks_bar_on_a_terminal(self):
        class Tty(io.StringIO):
            def isatty(self):
                return True

        assert resolve_progress_style("auto", Tty()) == "bar"

    def test_auto_picks_bar_under_pycharm(self, monkeypatch):
        """PyCharm's run console is a pipe but renders carriage returns."""
        monkeypatch.delenv(PROGRESS_ENV_VAR, raising=False)
        monkeypatch.setenv("PYCHARM_HOSTED", "1")
        assert resolve_progress_style("auto", io.StringIO()) == "bar"

    def test_auto_falls_back_to_lines_when_redirected(self, monkeypatch):
        monkeypatch.delenv("PYCHARM_HOSTED", raising=False)
        monkeypatch.delenv(PROGRESS_ENV_VAR, raising=False)
        assert resolve_progress_style("auto", io.StringIO()) == "lines"

    def test_env_var_overrides_auto(self, monkeypatch):
        monkeypatch.setenv(PROGRESS_ENV_VAR, "bar")
        assert resolve_progress_style("auto", io.StringIO()) == "bar"

    def test_explicit_style_beats_the_env_var(self, monkeypatch):
        monkeypatch.setenv(PROGRESS_ENV_VAR, "bar")
        assert resolve_progress_style("lines", io.StringIO()) == "lines"

    def test_unknown_style_rejected(self):
        with pytest.raises(ValueError, match="progress must be one of"):
            resolve_progress_style("fancy", io.StringIO())

    def test_ascii_glyphs_when_the_stream_cannot_encode_blocks(self):
        class Cp1252(io.StringIO):
            encoding = "cp1252"

        assert bar_glyphs(Cp1252()) == ("#", "-")

    def test_block_glyphs_on_utf8(self):
        class Utf8(io.StringIO):
            encoding = "utf-8"

        assert bar_glyphs(Utf8()) == ("█", "░")

    def test_eta_ignores_work_inherited_from_a_resume(self, tmp_path, capsys):
        """Skipped tiles took no time now; counting them would deflate the ETA.

        The bar must still show them as done - the run really is that far
        along - but the throughput used to extrapolate has to come from
        tiles this run actually computed.
        """
        image = branching_image((96, 128))
        source = tmp_path / "img.npy"
        np.save(source, image)

        skeletonize_streaming(source, tmp_path, tile_shape=32, halo=8)
        log = tmp_path / DONE_LOG_NAME
        log.write_text("\n".join(log.read_text().strip().splitlines()[:8]) + "\n")
        capsys.readouterr()

        skeletonize_streaming(source, tmp_path, tile_shape=32, halo=8, progress="lines")

        lines = capsys.readouterr().out.splitlines()
        # The skipped tiles are credited to the bar, but nothing has been
        # measured yet, so the rate is unknown rather than fabricated.
        credited = next(line for line in lines if "8/12 tiles" in line)
        assert credited.rstrip().endswith("ETA ?")
        # Voxel-weighted, so 8 of 12 tiles is 68.8% of the work, not 66.7%.
        assert "68.8%" in credited
        assert "12/12 tiles" in lines[-2]

    def test_eta_is_unknown_before_any_tile_completes(self, tmp_path, capsys):
        skeletonize_streaming(
            np.zeros((48, 48), dtype=np.uint8),
            tmp_path,
            tile_shape=16,
            halo=4,
            progress="lines",
        )
        assert capsys.readouterr().out.splitlines()[0].rstrip().endswith("ETA ?")

    def test_percentage_is_weighted_by_voxels_not_tile_count(self, tmp_path, capsys):
        """A 2-voxel-deep edge tile is not half the work of a full one."""
        # 2 tiles on axis 0: cores 0-16 (read 0-18) and 16-18 (read 14-18).
        grid = TileGrid(shape=(18, 16), tile_shape=16, halo=2)
        assert len(grid) == 2
        skeletonize_streaming(
            np.zeros(grid.shape, dtype=np.uint8),
            tmp_path,
            tile_shape=16,
            halo=2,
            progress="lines",
        )

        first = capsys.readouterr().out.splitlines()[1]
        assert "1/2 tiles" in first
        # 18*16 read of 18*16 + 4*16 total -> 81.8%, not the 50% a tile count
        # would report.
        assert "81.8%" in first


class TestManifest:
    def test_roundtrip(self, tmp_path):
        grid = TileGrid(shape=(70, 53), tile_shape=(16, 32), halo=5)
        write_manifest(grid, tmp_path)
        assert read_manifest(tmp_path) == grid

    def test_records_every_tile(self, tmp_path):
        grid = TileGrid(shape=(30, 40, 50), tile_shape=16, halo=3)
        write_manifest(grid, tmp_path)

        data = json.loads((tmp_path / MANIFEST_NAME).read_text())
        assert data["num_tiles"] == len(grid) == len(data["tiles"])
        assert data["read_overhead"] == pytest.approx(grid.read_overhead, abs=1e-4)

    def test_recorded_coordinates_match_the_specs(self, tmp_path):
        grid = TileGrid(shape=(30, 40, 50), tile_shape=16, halo=3)
        write_manifest(grid, tmp_path)

        records = json.loads((tmp_path / MANIFEST_NAME).read_text())["tiles"]
        for spec, record in zip(grid, records):
            assert tuple(record["index"]) == spec.index
            for key, slices in (
                ("core", spec.core),
                ("read", spec.read),
                ("core_in_read", spec.core_in_read),
            ):
                assert record[key]["start"] == [s.start for s in slices]
                assert record[key]["stop"] == [s.stop for s in slices]
            assert tuple(record["core_shape"]) == spec.core_shape
            assert tuple(record["read_shape"]) == spec.read_shape

    def test_records_no_per_tile_file_paths(self, tmp_path):
        """Streaming stages nothing, so the manifest must not promise files."""
        write_manifest(TileGrid(shape=(48, 48), tile_shape=16, halo=4), tmp_path)

        records = json.loads((tmp_path / MANIFEST_NAME).read_text())["tiles"]
        assert all("file" not in key for record in records for key in record)

    def test_coordinates_place_the_tile_back_in_the_volume(self, tmp_path):
        """The recorded coords alone must be enough to reassemble the volume."""
        rng = np.random.default_rng(0)
        volume = rng.integers(0, 2, size=(37, 41), dtype=np.uint8)
        write_manifest(TileGrid(shape=volume.shape, tile_shape=8, halo=3), tmp_path)

        def bounds(record, key):
            return tuple(
                slice(a, b)
                for a, b in zip(record[key]["start"], record[key]["stop"])
            )

        rebuilt = np.zeros(volume.shape, dtype=np.uint8)
        for record in json.loads((tmp_path / MANIFEST_NAME).read_text())["tiles"]:
            block = volume[bounds(record, "read")]
            assert block.shape == tuple(record["read_shape"])
            rebuilt[bounds(record, "core")] = block[bounds(record, "core_in_read")]

        assert np.array_equal(rebuilt, volume)

    def test_stale_tile_list_is_rejected(self, tmp_path):
        write_manifest(TileGrid(shape=(48, 48), tile_shape=16, halo=4), tmp_path)

        path = tmp_path / MANIFEST_NAME
        data = json.loads(path.read_text())
        data["tiles"] = data["tiles"][:-1]
        path.write_text(json.dumps(data))

        with pytest.raises(ValueError, match="lists 8 tiles"):
            read_manifest(tmp_path)


class TestOutputFile:
    def test_written_file_is_readable_afterwards(self, tmp_path):
        """The write mapping must be released, or Windows keeps it locked."""
        image = branching_image((48, 48))
        out = skeletonize_streaming(image, tmp_path, tile_shape=16, halo=4)

        assert np.load(out, mmap_mode="r").shape == image.shape
        # Rerunning over the same path must not hit a lock either.
        skeletonize_streaming(image, tmp_path, tile_shape=16, halo=4, resume=False)

    def test_honours_out_path(self, tmp_path):
        image = branching_image((48, 48))
        custom = skeletonize_streaming(
            image, tmp_path, tile_shape=16, halo=4, out_path=tmp_path / "n" / "s.npy"
        )
        assert custom == tmp_path / "n" / "s.npy"
        assert np.array_equal(np.load(custom), lee94_thin(image))

    def test_output_is_uint8_of_the_source_shape(self, tmp_path):
        volume = branching_volume()
        out = np.load(
            skeletonize_streaming(volume, tmp_path, tile_shape=16, halo=HALO_3D)
        )
        assert out.shape == volume.shape
        assert out.dtype == np.uint8


class TestSkeletonEquivalence:
    """Tiled thinning must reproduce monolithic thinning exactly."""

    def test_2d(self, tmp_path):
        image = branching_image((96, 128))
        grid = TileGrid(shape=image.shape, tile_shape=32, halo=8)
        # Guard against a vacuous test: tiles must genuinely see less than
        # the whole image, otherwise this proves nothing about the halo.
        assert any(s.read_shape != image.shape for s in grid)

        tiled = tiled_skeleton(image, tmp_path, tile_shape=32, halo=8)
        assert np.array_equal(tiled, lee94_thin(image))

    def test_3d(self, tmp_path):
        volume = branching_volume()
        grid = TileGrid(shape=volume.shape, tile_shape=16, halo=HALO_3D)
        assert any(s.read_shape != volume.shape for s in grid)

        tiled = tiled_skeleton(volume, tmp_path, tile_shape=16, halo=HALO_3D)
        assert np.array_equal(tiled, lee94_thin(volume))

    def test_anisotropic_tiles(self, tmp_path):
        image = branching_image((96, 128))
        tiled = tiled_skeleton(image, tmp_path, tile_shape=(32, 48), halo=8)
        assert np.array_equal(tiled, lee94_thin(image))

    def test_empty_image(self, tmp_path):
        image = np.zeros((48, 48), dtype=np.uint8)
        assert not tiled_skeleton(image, tmp_path, tile_shape=16, halo=4).any()

    def test_non_binary_source_is_binarised(self, tmp_path):
        """A 0/255 mask must be converted before thinning.

        `thin_3d` binarises internally but `thin_2d` expects strict 0/1 and
        silently returns the input untouched otherwise - so the conversion in
        the tiled path is load-bearing, not redundant. The reference here is
        `lee94_thin` on an already-binarised image, which is what
        `analyze_binary_image` feeds it.
        """
        image = branching_image((48, 48)) * 255
        tiled = tiled_skeleton(image, tmp_path, tile_shape=16, halo=4)
        assert tiled.any()
        assert np.array_equal(tiled, lee94_thin(to_binary(image)))

    def test_insufficient_halo_actually_differs(self, tmp_path):
        """The halo has to matter, or the equivalence tests above are luck."""
        image = np.zeros((64, 64), dtype=np.uint8)
        image[8:56, 20:44] = 1  # thick slab straddling the tile seam at 32

        correct = tiled_skeleton(image, tmp_path / "wide", tile_shape=32, halo=24)
        starved = tiled_skeleton(image, tmp_path / "none", tile_shape=32, halo=0)

        assert np.array_equal(correct, lee94_thin(image))
        assert not np.array_equal(starved, correct)

    def test_3d_halo_must_exceed_structure_thickness(self, tmp_path):
        """3D's six-direction sweep reaches ~4x further than the thickness.

        Regression guard for the halo sizing documented in `vesskel.tiling`:
        a halo of 4 is not enough for a 4-voxel-thick volume, 16 is.
        """
        volume = branching_volume()
        reference = lee94_thin(volume)

        starved = tiled_skeleton(volume, tmp_path / "thin", tile_shape=16, halo=4)
        adequate = tiled_skeleton(
            volume, tmp_path / "thick", tile_shape=16, halo=HALO_3D
        )

        assert not np.array_equal(starved, reference)
        assert np.array_equal(adequate, reference)


class TestAnalyzeTiled:
    """The config-aware entry point: thinning only, everything else rejected."""

    def test_matches_monolithic(self, tmp_path):
        image = branching_image((96, 128))
        config = plain_config()

        expected = analyze_binary_image(image, "ref", config)
        assert (
            analyze_tiled(
                image, "ref", config, work_dir=tmp_path, tile_shape=32, halo=8
            )
            is None
        )

        written = np.load(tmp_path / "ref_skeleton.npy")
        assert np.array_equal(written, expected.skeleton)

    def test_3d_matches_monolithic(self, tmp_path):
        volume = branching_volume()
        config = plain_config()

        expected = analyze_binary_image(volume, "vol", config)
        analyze_tiled(
            volume, "vol", config, work_dir=tmp_path, tile_shape=16, halo=HALO_3D
        )

        written = np.load(tmp_path / "vol_skeleton.npy")
        assert np.array_equal(written, expected.skeleton)

    def test_writes_skeleton_npy_named_after_base_name(self, tmp_path):
        image = branching_image((48, 48))
        analyze_tiled(
            image, "sample", plain_config(), work_dir=tmp_path, tile_shape=16, halo=4
        )

        written = tmp_path / "sample_skeleton.npy"
        assert written.is_file()
        skeleton = np.load(written)
        assert skeleton.shape == image.shape
        assert skeleton.dtype == np.uint8
        assert skeleton.any()

    def test_empty_image(self, tmp_path):
        analyze_tiled(
            np.zeros((48, 48), dtype=np.uint8),
            "empty",
            plain_config(),
            work_dir=tmp_path,
            tile_shape=16,
            halo=4,
        )
        assert not np.load(tmp_path / "empty_skeleton.npy").any()

    def test_source_is_never_mutated(self, tmp_path):
        """Blocks are binarised in place, so `read` must hand back a copy.

        A single tile covering the whole source would otherwise alias it, and
        the in-place binarisation would rewrite the caller's array.
        """
        image = branching_image((48, 48)) * 255
        analyze_tiled(
            image, "src", plain_config(), work_dir=tmp_path, tile_shape=64, halo=4
        )
        assert set(np.unique(image)) == {0, 255}
        assert np.load(tmp_path / "src_skeleton.npy").any()

    def test_read_only_source_works(self, tmp_path):
        """A tifffile memmap is read-only; per-block copies must handle that."""
        image = branching_image((48, 48))
        image.flags.writeable = False

        analyze_tiled(
            image, "ro", plain_config(), work_dir=tmp_path, tile_shape=16, halo=4
        )
        assert np.array_equal(np.load(tmp_path / "ro_skeleton.npy"), lee94_thin(image))

    def test_writes_no_per_tile_files(self, tmp_path):
        """The streaming path goes source -> output; nothing is staged."""
        image = branching_image((48, 48))
        analyze_tiled(
            image, "keep", plain_config(), work_dir=tmp_path, tile_shape=16, halo=4
        )

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            DONE_LOG_NAME,
            "keep_skeleton.npy",
            MANIFEST_NAME,
        ]

    @pytest.mark.parametrize(
        "extraction",
        [
            {"closing_iterations": 1},
            {"fill_holes": True},
            {"max_hole_size": 64},
            {"junction_cleanup": True},
        ],
    )
    def test_unsupported_steps_are_rejected(self, tmp_path, extraction):
        """Silently skipping these would diverge from the monolithic path."""
        (name,) = extraction
        with pytest.raises(NotImplementedError, match=name):
            analyze_tiled(
                branching_image((48, 48)),
                "nope",
                plain_config(**extraction),
                work_dir=tmp_path,
                tile_shape=16,
                halo=4,
            )
        assert not (tmp_path / "nope_skeleton.npy").exists()

    def test_error_names_every_configured_step(self, tmp_path):
        with pytest.raises(NotImplementedError) as excinfo:
            analyze_tiled(
                branching_image((48, 48)),
                "nope",
                plain_config(fill_holes=True, junction_cleanup=True),
                work_dir=tmp_path,
                tile_shape=16,
                halo=4,
            )
        message = str(excinfo.value)
        assert "fill_holes" in message and "junction_cleanup" in message

    def test_extraction_and_output_settings_are_simply_unused(self, tmp_path):
        """Nothing past the skeleton is produced, so these are not errors.

        `summary` in particular defaults to True - rejecting it would make a
        default config unusable.
        """
        image = branching_image((48, 48))
        analyze_tiled(
            image,
            "feat",
            plain_config(branches=True, nodes=True, summary=True, vessel_radius=True),
            work_dir=tmp_path,
            tile_shape=16,
            halo=4,
        )
        assert np.array_equal(
            np.load(tmp_path / "feat_skeleton.npy"), lee94_thin(image)
        )


class TestStreaming:
    """Reading blocks from the source file instead of loading it."""

    def test_matches_monolithic_from_a_npy_path(self, tmp_path):
        volume = branching_volume()
        source = tmp_path / "vol.npy"
        np.save(source, volume)

        out = skeletonize_streaming(
            source, tmp_path / "work", tile_shape=16, halo=HALO_3D
        )
        assert np.array_equal(np.load(out), lee94_thin(volume))

    def test_matches_monolithic_from_a_tif_path(self, tmp_path):
        import tifffile

        volume = branching_volume()
        source = tmp_path / "vol.tif"
        tifffile.imwrite(source, volume)

        out = skeletonize_streaming(
            source, tmp_path / "work", tile_shape=16, halo=HALO_3D
        )
        assert np.array_equal(np.load(out), lee94_thin(volume))

    def test_source_file_is_never_fully_loaded(self, tmp_path):
        """A memmapped read must not materialise the whole array."""
        source = tmp_path / "vol.npy"
        np.save(source, branching_volume())

        reader = BlockReader(source)
        assert isinstance(reader.array, np.memmap)
        assert reader.shape == (32, 40, 48)

    def test_read_returns_an_owned_writeable_copy(self, tmp_path):
        """Blocks get binarised in place, so they must not alias the source."""
        volume = branching_volume()
        reader = BlockReader(volume)
        whole = (slice(None),) * volume.ndim

        block = reader.read(whole)
        assert block.flags.writeable and block.flags.c_contiguous
        block[:] = 0
        assert volume.any()

    def test_parallel_matches_serial(self, tmp_path):
        """Disjoint cores mean concurrent workers can share one output map."""
        source = tmp_path / "img.npy"
        np.save(source, branching_image((96, 128)))

        serial = np.load(
            skeletonize_streaming(source, tmp_path / "s", tile_shape=32, halo=8)
        )
        parallel = np.load(
            skeletonize_streaming(source, tmp_path / "p", tile_shape=32, halo=8, jobs=3)
        )
        assert np.array_equal(serial, parallel)
        assert serial.any()

    def test_rejects_4d_input(self, tmp_path):
        source = tmp_path / "hyper.npy"
        np.save(source, np.zeros((2, 2, 2, 2), dtype=np.uint8))
        with pytest.raises(ValueError, match="2D or 3D"):
            skeletonize_streaming(source, tmp_path / "work", tile_shape=2)

    def test_rejects_negative_jobs(self, tmp_path):
        with pytest.raises(ValueError, match="jobs must be"):
            skeletonize_streaming(
                branching_image((48, 48)), tmp_path, tile_shape=16, halo=4, jobs=-1
            )


class TestDoneLog:
    def test_records_one_line_per_tile(self, tmp_path):
        grid_shape = (48, 48)
        skeletonize_streaming(
            branching_image(grid_shape), tmp_path, tile_shape=16, halo=4
        )

        grid = TileGrid(shape=grid_shape, tile_shape=16, halo=4)
        lines = (tmp_path / DONE_LOG_NAME).read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]
        assert len(records) == len(grid)
        assert {tuple(r["index"]) for r in records} == {s.index for s in grid}
        # The index is the whole record: it is all a resume needs.
        assert all(set(r) == {"index"} for r in records)

    def test_resume_skips_finished_tiles_and_keeps_their_output(self, tmp_path):
        image = branching_image((96, 128))
        source = tmp_path / "img.npy"
        np.save(source, image)

        out = skeletonize_streaming(source, tmp_path, tile_shape=32, halo=8)
        complete = np.load(out).copy()

        # Simulate a crash after 4 of the 12 tiles: truncate the log, and
        # blank the output where those unfinished tiles had written.
        log = tmp_path / DONE_LOG_NAME
        kept = log.read_text().strip().splitlines()[:4]
        log.write_text("\n".join(kept) + "\n")
        done = {tuple(json.loads(line)["index"]) for line in kept}

        scratch = np.load(out, mmap_mode="r+")
        for spec in TileGrid(shape=image.shape, tile_shape=32, halo=8):
            if spec.index not in done:
                scratch[spec.core] = 0
        scratch.flush()
        del scratch

        skeletonize_streaming(source, tmp_path, tile_shape=32, halo=8)

        assert np.array_equal(np.load(out), complete)
        assert len((tmp_path / DONE_LOG_NAME).read_text().strip().splitlines()) == 12

    def test_a_changed_grid_discards_the_old_log(self, tmp_path):
        image = branching_image((96, 128))
        skeletonize_streaming(image, tmp_path, tile_shape=32, halo=8)
        assert len((tmp_path / DONE_LOG_NAME).read_text().splitlines()) == 12

        skeletonize_streaming(image, tmp_path, tile_shape=48, halo=8)

        grid = TileGrid(shape=image.shape, tile_shape=48, halo=8)
        lines = (tmp_path / DONE_LOG_NAME).read_text().strip().splitlines()
        assert len(lines) == len(grid)
        assert np.array_equal(np.load(tmp_path / "skeleton.npy"), lee94_thin(image))

    def test_resume_false_redoes_everything(self, tmp_path):
        image = branching_image((48, 48))
        skeletonize_streaming(image, tmp_path, tile_shape=16, halo=4)
        first = (tmp_path / DONE_LOG_NAME).read_text()

        skeletonize_streaming(image, tmp_path, tile_shape=16, halo=4, resume=False)
        assert (tmp_path / DONE_LOG_NAME).read_text() == first

    def test_truncated_final_line_is_ignored(self, tmp_path):
        image = branching_image((48, 48))
        skeletonize_streaming(image, tmp_path, tile_shape=16, halo=4)

        log = tmp_path / DONE_LOG_NAME
        log.write_text(log.read_text() + '{"index": [2, ')  # killed mid-write

        skeletonize_streaming(image, tmp_path, tile_shape=16, halo=4)
        assert np.array_equal(np.load(tmp_path / "skeleton.npy"), lee94_thin(image))

    def test_missing_log_reads_as_empty(self, tmp_path):
        assert DoneLog(tmp_path).read() == set()

    def test_record_outside_a_with_block_raises(self, tmp_path):
        spec = next(iter(TileGrid(shape=(8, 8), tile_shape=8, halo=0)))
        with pytest.raises(RuntimeError, match="outside of a `with` block"):
            DoneLog(tmp_path).record(spec)
