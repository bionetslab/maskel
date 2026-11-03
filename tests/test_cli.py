import json
import subprocess
import sys

import h5py
import numpy as np
import pytest

from maskel.config import CONFIG_SCHEMA_VERSION

HEAVY_MODULES = frozenset(
    {"numpy", "PIL", "PIL.Image", "tifffile", "maskel.pipeline", "maskel.io"}
)


def _main(monkeypatch, *argv) -> int:
    from maskel.cli import main

    monkeypatch.setattr(sys, "argv", ["maskel", *map(str, argv)])
    return main()


def _write_mask(path):
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[22:26, 4:44] = 255
    mask[4:44, 22:26] = 255
    with h5py.File(path, "w") as f:
        f.create_dataset("mask", data=mask, chunks=(10, 10), compression="gzip")


class TestCompletionSpeed:
    """Shell completions must exit before importing heavy modules (numpy/PIL)."""

    def test_completions_skip_heavy_imports(self):
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

heavy = [m for m in {sorted(HEAVY_MODULES)!r} if m in sys.modules]
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
        assert result.returncode == 0, result.stderr
        assert "HEAVY:" not in result.stdout + result.stderr


class TestConfigCommands:
    def test_init_then_validate(self, tmp_path, monkeypatch):
        path = tmp_path / "cfg.json"
        assert _main(monkeypatch, "init", path) == 0
        assert json.loads(path.read_text())["schema_version"] == CONFIG_SCHEMA_VERSION
        assert _main(monkeypatch, "validate", path) == 0

    def test_version_flag(self, capsys, monkeypatch):
        with pytest.raises(SystemExit) as exc:
            _main(monkeypatch, "--version")
        assert exc.value.code == 0
        assert "maskel" in capsys.readouterr().out


class TestStageCommands:
    def test_run_writes_everything(self, tmp_path, monkeypatch):
        _write_mask(tmp_path / "cross.h5")
        argv = ["run", "--input", tmp_path / "cross.h5", "--out", tmp_path / "out"]
        argv += ["--tile-shape", "24", "--halo", "16"]
        assert _main(monkeypatch, *argv) == 0
        names = {p.name for p in (tmp_path / "out").iterdir()}
        assert names == {
            "cross_voxels.h5",
            "cross_voxels_post.h5",
            "cross_summary.csv",
        }

    def test_stages_one_at_a_time(self, tmp_path, monkeypatch):
        _write_mask(tmp_path / "cross.h5")
        out = tmp_path / "out"
        common = ["--out", out]
        assert (
            _main(
                monkeypatch,
                "skeletonize",
                "--input",
                tmp_path / "cross.h5",
                *common,
            )
            == 0
        )
        assert (
            _main(
                monkeypatch, "postprocess", "--input", out / "cross_voxels.h5", *common
            )
            == 0
        )
        assert (
            _main(
                monkeypatch,
                "features",
                "--input",
                out / "cross_voxels_post.h5",
                *common,
            )
            == 0
        )
        assert (out / "cross_summary.csv").is_file()

    def test_spacing_comes_from_config(self, tmp_path, monkeypatch):
        from maskel.config import PipelineConfig, save_pipeline_config
        from maskel.io.voxels import read_voxels

        _write_mask(tmp_path / "cross.h5")
        save_pipeline_config(PipelineConfig(spacing=(2.0, 0.5)), tmp_path / "cfg.json")
        argv = ["skeletonize", "--input", tmp_path / "cross.h5", "--out", tmp_path]
        argv += ["--config", tmp_path / "cfg.json"]
        assert _main(monkeypatch, *argv) == 0
        assert read_voxels(tmp_path / "cross_voxels.h5").spacing == (2.0, 0.5)
