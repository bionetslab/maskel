import json

import pytest

from maskel.config import (
    CONFIG_SCHEMA_VERSION,
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    load_pipeline_config,
    save_pipeline_config,
)


class TestExtractionConfig:
    def test_defaults(self):
        c = ExtractionConfig()
        assert c.branches is False
        assert c.branch_text is False
        assert c.summary is True
        assert c.fractal_dimension is False
        assert c.vessel_radius is False

    def test_custom_values(self):
        c = ExtractionConfig(
            branches=False,
            branch_text=False,
            summary=False,
            fractal_dimension=True,
            vessel_radius=True,
        )
        assert c.branches is False
        assert c.branch_text is False
        assert c.summary is False
        assert c.fractal_dimension is True
        assert c.vessel_radius is True

    def test_round_trip_dict(self):
        original = ExtractionConfig(
            branches=True,
            branch_text=False,
            summary=True,
            fractal_dimension=True,
            vessel_radius=False,
        )
        restored = ExtractionConfig.from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_defaults_on_missing_keys(self):
        c = ExtractionConfig.from_dict({})
        assert c == ExtractionConfig()

    def test_from_dict_warns_on_unknown_keys(self, capsys):
        ExtractionConfig.from_dict({"branches": True, "foo": 1, "bar": 2})
        captured = capsys.readouterr()
        assert "ignored unknown keys" in captured.err
        assert "'bar'" in captured.err
        assert "'foo'" in captured.err


class TestOutputConfig:
    def test_defaults(self):
        c = OutputConfig()
        assert c.write_skeleton_npy is True
        assert c.write_skeleton_png is False
        assert c.write_summary_csv is True
        assert c.write_branch_csv is False
        assert c.write_radius is False
        assert c.write_graphml is False

    def test_round_trip_dict(self):
        original = OutputConfig(
            write_skeleton_npy=False,
            write_skeleton_png=True,
            write_summary_csv=True,
            write_branch_csv=True,
            write_radius=True,
            write_graphml=True,
        )
        restored = OutputConfig.from_dict(original.to_dict())
        assert restored == original

    def test_from_none_defaults(self):
        c = OutputConfig.from_dict(None)
        assert c == OutputConfig()

    def test_from_empty_dict_defaults(self):
        c = OutputConfig.from_dict({})
        assert c == OutputConfig()

    def test_from_dict_coerces_bools(self):
        c = OutputConfig.from_dict({"write_summary_csv": 0, "write_skeleton_png": 1})
        assert c.write_summary_csv is False
        assert c.write_skeleton_png is True

        c = OutputConfig.from_dict({"write_summary_csv": 1, "write_skeleton_png": 0})
        assert c.write_summary_csv is True
        assert c.write_skeleton_png is False

    def test_from_dict_falls_back_on_falsy(self):
        c = OutputConfig.from_dict(0)
        assert c == OutputConfig()
        c = OutputConfig.from_dict("")
        assert c == OutputConfig()

    def test_from_dict_warns_on_unknown_keys(self, capsys):
        OutputConfig.from_dict({"write_summary_csv": True, "nope": 99})
        captured = capsys.readouterr()
        assert "ignored unknown keys" in captured.err
        assert "'nope'" in captured.err


class TestPipelineConfig:
    def test_round_trip_dict(self):
        original = PipelineConfig(
            extraction=ExtractionConfig(fractal_dimension=True, vessel_radius=True),
            output=OutputConfig(write_branch_csv=True, write_radius=True),
        )
        as_dict = original.to_dict()
        restored = PipelineConfig.from_dict(as_dict)
        assert restored == original

    def test_default_schema_version(self):
        c = PipelineConfig(
            extraction=ExtractionConfig(),
            output=OutputConfig(),
        )
        assert c.schema_version == CONFIG_SCHEMA_VERSION

    def test_from_dict_preserves_schema_version_in_object(self):
        data = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "extraction": {},
            "output": {},
        }
        c = PipelineConfig.from_dict(data)
        assert c.schema_version == CONFIG_SCHEMA_VERSION

    def test_from_dict_unsupported_schema_version(self):
        data = {
            "schema_version": 999,
            "extraction": {},
            "output": {},
        }
        with pytest.raises(ValueError, match="Unsupported schema_version"):
            PipelineConfig.from_dict(data)

    def test_from_dict_rejects_none(self):
        with pytest.raises(TypeError, match="must be an object"):
            PipelineConfig.from_dict(None)

    def test_from_dict_rejects_non_dict(self):
        with pytest.raises(TypeError, match="must be an object"):
            PipelineConfig.from_dict(["not", "a", "dict"])

    def test_from_dict_rejects_non_dict_extraction(self):
        data = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "extraction": "bad",
            "output": {},
        }
        with pytest.raises(TypeError, match="'extraction' must be an object"):
            PipelineConfig.from_dict(data)

    def test_from_dict_rejects_non_dict_output(self):
        data = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "extraction": {},
            "output": 123,
        }
        with pytest.raises(TypeError, match="'output' must be an object"):
            PipelineConfig.from_dict(data)

    def test_from_dict_missing_schema_version_defaults(self):
        data = {"extraction": {}, "output": {}}
        c = PipelineConfig.from_dict(data)
        assert c.schema_version == CONFIG_SCHEMA_VERSION

    def test_to_dict_structure(self):
        c = PipelineConfig(
            extraction=ExtractionConfig(),
            output=OutputConfig(),
        )
        d = c.to_dict()
        assert "schema_version" in d
        assert "extraction" in d
        assert "output" in d
        assert isinstance(d["extraction"], dict)
        assert isinstance(d["output"], dict)

    def test_from_dict_warns_on_unknown_keys(self, capsys):
        data = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "extraction": {},
            "output": {},
            "unknown": "should warn",
        }
        PipelineConfig.from_dict(data)
        captured = capsys.readouterr()
        assert "ignored unknown keys" in captured.err
        assert "'unknown'" in captured.err


class TestConfigFileIO:
    def test_save_and_load_round_trip(self, tmp_path):
        config = PipelineConfig(
            extraction=ExtractionConfig(vessel_radius=True),
            output=OutputConfig(write_skeleton_png=True, write_radius=True),
        )
        path = tmp_path / "config.json"
        save_pipeline_config(config, path)

        loaded = load_pipeline_config(path)
        assert loaded == config

    def test_save_creates_valid_json(self, tmp_path):
        config = PipelineConfig(
            extraction=ExtractionConfig(),
            output=OutputConfig(),
        )
        path = tmp_path / "pipeline.json"
        save_pipeline_config(config, path)

        with path.open(encoding="utf-8") as f:
            raw = json.load(f)

        assert raw["schema_version"] == CONFIG_SCHEMA_VERSION
        assert raw["extraction"]["branches"] is False
        assert raw["output"]["write_summary_csv"] is True

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline_config(tmp_path / "nonexistent.json")

    def test_load_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_pipeline_config(path)

    def test_save_to_string_path(self, tmp_path):
        config = PipelineConfig(
            extraction=ExtractionConfig(),
            output=OutputConfig(),
        )
        path = str(tmp_path / "str_save.json")
        save_pipeline_config(config, path)
        loaded = load_pipeline_config(path)
        assert loaded == config

    def test_load_from_string_path(self, tmp_path):
        config = PipelineConfig(
            extraction=ExtractionConfig(),
            output=OutputConfig(),
        )
        path = tmp_path / "str_config.json"
        save_pipeline_config(config, path)
        loaded = load_pipeline_config(str(path))
        assert loaded == config
