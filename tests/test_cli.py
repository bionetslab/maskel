import argparse
import csv
import json

import numpy as np
import pytest
from PIL import Image

from maskel._io import (
    load_image,
    sanitize_for_csv,
    save_radius,
    save_skeleton,
    write_csv,
)
from maskel.cli import _discover_input_paths
from maskel.config import CONFIG_SCHEMA_VERSION, PipelineConfig

HEAVY_MODULES = frozenset(
    {"numpy", "PIL", "PIL.Image", "maskel.pipeline", "maskel._batch", "maskel._io"}
)


class TestCompletionSpeed:
    """Shell completions must exit before importing heavy modules (numpy/PIL)."""

    def test_completions_skip_heavy_imports(self):
        import subprocess
        import sys

        heavy_modules = sorted(HEAVY_MODULES)
        probe = f"""
import os, sys

def _fake_exit(code=0):
    raise SystemExit(code)
os._exit = _fake_exit

os.environ["_ARGCOMPLETE"] = "1"
sys.argv = ["maskel", "complete", "zsh"]

try:
    import maskel.cli
except SystemExit:
    pass

heavy = [m for m in {heavy_modules!r} if m in sys.modules]
if heavy:
    sys.stdout.write("HEAVY:" + ",".join(heavy))
"""
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert result.returncode == 0, f"Subprocess failed (stderr): {result.stderr}"
        assert "HEAVY:" not in result.stdout, (
            f"Heavy modules loaded during completions: {result.stdout}"
        )
        assert "HEAVY:" not in result.stderr, (
            f"Heavy modules leaked to stderr: {result.stderr}"
        )


class TestDiscoverInputPaths:
    def test_single_png_file(self, tmp_path):
        (tmp_path / "a.png").touch()
        result = _discover_input_paths([str(tmp_path / "a.png")], recursive=False)
        assert len(result) == 1
        assert result[0].name == "a.png"

    def test_multiple_files(self, tmp_path):
        (tmp_path / "a.png").touch()
        (tmp_path / "b.jpg").touch()
        (tmp_path / "c.tif").touch()
        result = _discover_input_paths(
            [str(tmp_path / "a.png"), str(tmp_path / "b.jpg"), str(tmp_path / "c.tif")],
            recursive=False,
        )
        assert len(result) == 3

    def test_directory_non_recursive(self, tmp_path):
        (tmp_path / "a.png").touch()
        (tmp_path / "b.jpg").touch()
        (tmp_path / "not_an_image.txt").touch()
        result = _discover_input_paths([str(tmp_path)], recursive=False)
        assert len(result) == 2

    def test_directory_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "root.png").touch()
        (sub / "nested.tif").touch()
        (sub / "skip.txt").touch()
        result = _discover_input_paths([str(tmp_path)], recursive=True)
        names = {p.name for p in result}
        assert "root.png" in names
        assert "nested.tif" in names
        assert "skip.txt" not in names

    def test_npy_file_supported(self, tmp_path):
        (tmp_path / "vol.npy").touch()
        result = _discover_input_paths([str(tmp_path / "vol.npy")], recursive=False)
        assert len(result) == 1

    def test_glob_pattern(self, tmp_path):
        (tmp_path / "image_01.png").touch()
        (tmp_path / "image_02.png").touch()
        (tmp_path / "other.jpg").touch()
        result = _discover_input_paths([str(tmp_path / "image_*.png")], recursive=False)
        assert len(result) == 2


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

    def test_load_invalid_dimension_raises(self, tmp_path):
        path = tmp_path / "bad.npy"
        np.save(path, np.ones((2, 2, 2, 2), dtype=np.uint8))
        with pytest.raises(
            ValueError, match=r"Expected 2D or 3D image, got shape=\(2, 2, 2, 2\)"
        ):
            load_image(path)


class TestSanitizeForCsv:
    def test_numpy_int(self):
        assert sanitize_for_csv(np.int64(42)) == 42

    def test_numpy_float(self):
        result = sanitize_for_csv(np.float64(3.14))
        assert isinstance(result, float)
        assert result == 3.14

    def test_python_int_passes_through(self):
        assert sanitize_for_csv(42) == 42

    def test_python_str_passes_through(self):
        assert sanitize_for_csv("hello") == "hello"

    def test_none_passes_through(self):
        assert sanitize_for_csv(None) is None

    def test_bool_passes_through(self):
        assert sanitize_for_csv(True) is True
        assert sanitize_for_csv(False) is False

    def test_list_passes_through(self):
        assert sanitize_for_csv([1, "a"]) == [1, "a"]


class TestWriteCsv:
    def test_basic_write(self, tmp_path):
        path = tmp_path / "out.csv"
        rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        write_csv(path, rows)
        with path.open() as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 2
        assert data[0]["a"] == "1"
        assert data[1]["b"] == "4"

    def test_empty_rows_writes_nothing(self, tmp_path):
        path = tmp_path / "empty.csv"
        write_csv(path, [])
        assert not path.exists()


class TestSaveSkeleton:
    def test_save_npy(self, tmp_path):
        path = tmp_path / "skel"
        skeleton = np.eye(10, dtype=np.uint8)
        save_skeleton(path, skeleton, npy=True, png=False)
        loaded = np.load(path.with_suffix(".npy"))
        assert np.array_equal(loaded, skeleton)

    def test_save_png_2d(self, tmp_path):
        path = tmp_path / "skel"
        skeleton = np.eye(8, dtype=np.uint8)
        save_skeleton(path, skeleton, npy=False, png=True)
        loaded = np.array(Image.open(path.with_suffix(".png")))
        assert loaded.shape == (8, 8)
        assert np.array_equal(loaded > 0, skeleton > 0)

    def test_save_png_3d_warns_and_skips(self, tmp_path, capsys):
        path = tmp_path / "skel"
        skeleton = np.eye(4, dtype=np.uint8).reshape(4, 1, 4)
        save_skeleton(path, skeleton, npy=True, png=True)
        assert not path.with_suffix(".png").exists()
        assert path.with_suffix(".npy").exists()
        assert "2D" in capsys.readouterr().err

    def test_save_both_formats(self, tmp_path):
        path = tmp_path / "skel"
        skeleton = np.eye(6, dtype=np.uint8)
        save_skeleton(path, skeleton, npy=True, png=True)
        assert path.with_suffix(".npy").exists()
        assert path.with_suffix(".png").exists()
        npy_loaded = np.load(path.with_suffix(".npy"))
        png_loaded = np.array(Image.open(path.with_suffix(".png")))
        assert np.array_equal(png_loaded > 0, npy_loaded > 0)

    def test_save_neither_does_nothing(self, tmp_path):
        path = tmp_path / "skel"
        save_skeleton(path, np.eye(3, dtype=np.uint8), npy=False, png=False)
        assert not path.with_suffix(".npy").exists()
        assert not path.with_suffix(".png").exists()

    def test_npy_output_is_uint8(self, tmp_path):
        path = tmp_path / "skel"
        skeleton = np.ones((4, 4), dtype=np.int32)
        save_skeleton(path, skeleton, npy=True, png=False)
        loaded = np.load(path.with_suffix(".npy"))
        assert loaded.dtype == np.uint8


class TestSaveRadius:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "radius"
        radius = np.random.default_rng(7).random((4, 4))
        save_radius(path, radius)
        loaded = np.load(path.with_suffix(".npy"))
        np.testing.assert_array_equal(loaded, radius)

    def test_saved_as_float64(self, tmp_path):
        path = tmp_path / "radius"
        radius = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float64)
        save_radius(path, radius)
        loaded = np.load(path.with_suffix(".npy"))
        assert loaded.dtype == np.float64


def _make_args(**kwargs):
    return argparse.Namespace(**kwargs)


class TestMainCommands:
    def test_config_init_creates_valid_json(self, tmp_path):
        from maskel.cli import _config_init

        args = _make_args(out=str(tmp_path / "config.json"))
        exit_code = _config_init(args)
        assert exit_code == 0

        with open(args.out) as f:
            data = json.load(f)
        assert data["schema_version"] == CONFIG_SCHEMA_VERSION
        assert "extraction" in data
        assert "output" in data

    def test_validate_config_succeeds(self, tmp_path):
        from maskel.cli import _validate_config

        config_path = tmp_path / "cfg.json"
        config = PipelineConfig.from_dict(
            {"schema_version": CONFIG_SCHEMA_VERSION, "extraction": {}, "output": {}}
        )
        config_path.write_text(json.dumps(config.to_dict(), indent=2))

        args = _make_args(config=str(config_path))
        exit_code = _validate_config(args)
        assert exit_code == 0

    def test_run_batch_processes_single_image(self, tmp_path):
        from maskel.cli import _run_batch

        img_path = tmp_path / "input"
        img_path.mkdir()
        image = np.zeros((16, 16), dtype=np.uint8)
        image[8, 4:12] = 1
        image[4:12, 8] = 1
        Image.fromarray(image * 255).save(img_path / "cross.png")

        config_path = tmp_path / "pipeline.json"
        config = PipelineConfig.from_dict(
            {"schema_version": CONFIG_SCHEMA_VERSION, "extraction": {}, "output": {}}
        )
        config_path.write_text(json.dumps(config.to_dict(), indent=2))

        out_dir = tmp_path / "output"

        args = _make_args(
            input=[str(img_path)],
            config=str(config_path),
            out=str(out_dir),
            recursive=False,
            jobs=1,
        )

        exit_code = _run_batch(args)
        assert exit_code == 0
        assert (out_dir / "cross").is_dir()
        skeleton_path = out_dir / "cross" / "cross_skeleton.npy"
        assert skeleton_path.exists()
        loaded = np.load(skeleton_path)
        assert np.count_nonzero(loaded) > 0
        assert (out_dir / "summary.csv").exists()

    def test_run_batch_no_input_files_raises(self, tmp_path):
        from maskel.cli import _run_batch

        config_path = tmp_path / "cfg.json"
        config = PipelineConfig.from_dict(
            {"schema_version": CONFIG_SCHEMA_VERSION, "extraction": {}, "output": {}}
        )
        config_path.write_text(json.dumps(config.to_dict(), indent=2))

        (tmp_path / "empty_dir").mkdir()
        args = _make_args(
            input=[str(tmp_path / "empty_dir")],
            config=str(config_path),
            out=str(tmp_path / "out"),
            recursive=False,
            jobs=1,
        )

        with pytest.raises(ValueError, match="No input files found"):
            _run_batch(args)

    def test_run_batch_parallel_multiple_images(self, tmp_path):
        from maskel.cli import _run_batch

        img_path = tmp_path / "input"
        img_path.mkdir()
        for name in ("a", "b"):
            image = np.zeros((16, 16), dtype=np.uint8)
            image[8, 4:12] = 1
            image[4:12, 8] = 1
            Image.fromarray(image * 255).save(img_path / f"{name}.png")

        config_path = tmp_path / "pipeline.json"
        config = PipelineConfig.from_dict(
            {"schema_version": CONFIG_SCHEMA_VERSION, "extraction": {}, "output": {}}
        )
        config_path.write_text(json.dumps(config.to_dict(), indent=2))

        out_dir = tmp_path / "output"

        args = _make_args(
            input=[str(img_path)],
            config=str(config_path),
            out=str(out_dir),
            recursive=False,
            jobs=2,
        )

        exit_code = _run_batch(args)
        assert exit_code == 0
        assert (out_dir / "a").is_dir()
        assert (out_dir / "b").is_dir()
        assert (out_dir / "a" / "a_skeleton.npy").exists()
        assert (out_dir / "b" / "b_skeleton.npy").exists()
        assert (out_dir / "summary.csv").exists()

        with open(out_dir / "summary.csv") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        assert len(data) == 2

    def test_run_batch_parallel_failure_raises(self, tmp_path):
        from maskel.cli import _run_batch

        img_path = tmp_path / "input"
        img_path.mkdir()
        image = np.zeros((16, 16), dtype=np.uint8)
        image[8, 4:12] = 1
        image[4:12, 8] = 1
        Image.fromarray(image * 255).save(img_path / "good.png")
        np.save(img_path / "bad.npy", np.zeros((2, 2, 2, 2), dtype=np.uint8))

        config_path = tmp_path / "pipeline.json"
        config = PipelineConfig.from_dict(
            {"schema_version": CONFIG_SCHEMA_VERSION, "extraction": {}, "output": {}}
        )
        config_path.write_text(json.dumps(config.to_dict(), indent=2))

        out_dir = tmp_path / "output"

        args = _make_args(
            input=[str(img_path)],
            config=str(config_path),
            out=str(out_dir),
            recursive=False,
            jobs=2,
        )

        with pytest.raises(RuntimeError, match=r"image\(s\) failed"):
            _run_batch(args)

    def test_run_batch_jobs_zero_auto_detect(self, tmp_path):
        from maskel.cli import _run_batch

        img_path = tmp_path / "input"
        img_path.mkdir()
        image = np.zeros((16, 16), dtype=np.uint8)
        image[8, 4:12] = 1
        image[4:12, 8] = 1
        Image.fromarray(image * 255).save(img_path / "cross.png")

        config_path = tmp_path / "pipeline.json"
        config = PipelineConfig.from_dict(
            {"schema_version": CONFIG_SCHEMA_VERSION, "extraction": {}, "output": {}}
        )
        config_path.write_text(json.dumps(config.to_dict(), indent=2))

        out_dir = tmp_path / "output"

        args = _make_args(
            input=[str(img_path)],
            config=str(config_path),
            out=str(out_dir),
            recursive=False,
            jobs=0,
        )

        exit_code = _run_batch(args)
        assert exit_code == 0
        assert (out_dir / "cross" / "cross_skeleton.npy").exists()

    def test_main_dispatch_config_init(self, tmp_path, monkeypatch):
        import sys

        from maskel.cli import main

        monkeypatch.setattr(sys, "argv", ["maskel", "init", str(tmp_path / "cfg.json")])
        exit_code = main()
        assert exit_code == 0
        assert (tmp_path / "cfg.json").exists()

    def test_version_flag(self, capsys, monkeypatch):
        import sys

        from maskel.cli import main

        monkeypatch.setattr(sys, "argv", ["maskel", "--version"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "maskel" in captured.out

    def test_version_short_flag(self, capsys, monkeypatch):
        import sys

        from maskel.cli import main

        monkeypatch.setattr(sys, "argv", ["maskel", "-v"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "maskel" in captured.out

    def test_main_dispatch_validate_config(self, tmp_path, monkeypatch):
        import sys

        from maskel.cli import main

        config_path = tmp_path / "cfg.json"
        config = PipelineConfig.from_dict(
            {"schema_version": CONFIG_SCHEMA_VERSION, "extraction": {}, "output": {}}
        )
        config_path.write_text(json.dumps(config.to_dict(), indent=2))

        monkeypatch.setattr(sys, "argv", ["maskel", "validate", str(config_path)])
        exit_code = main()
        assert exit_code == 0
