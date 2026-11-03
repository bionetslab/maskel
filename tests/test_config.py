import json

import pytest

from maskel.config import (
    CONFIG_SCHEMA_VERSION,
    FeatureConfig,
    OutputConfig,
    PipelineConfig,
    PostprocessingConfig,
    PreprocessingConfig,
    load_pipeline_config,
    save_pipeline_config,
)

BLOCKS = [
    (PreprocessingConfig, {"opening_radius": 2, "fill_holes": True}),
    (PostprocessingConfig, {"prune_spurs": True, "min_spur_length": 4.0}),
    (FeatureConfig, {"branches": True, "radius_features": True}),
]


class TestBlocks:
    def test_defaults(self):
        assert PreprocessingConfig().opening_radius == 0
        assert PostprocessingConfig().cleanup_threshold_factor == 2.5
        assert FeatureConfig().summary is True
        assert FeatureConfig().radius_features is False

    @pytest.mark.parametrize(("cls", "values"), BLOCKS)
    def test_round_trip_dict(self, cls, values):
        original = cls(**values)
        assert cls.from_dict(original.to_dict()) == original

    @pytest.mark.parametrize(("cls", "values"), BLOCKS)
    def test_from_dict_defaults_on_missing_keys(self, cls, values):
        assert cls.from_dict({}) == cls()

    @pytest.mark.parametrize(("cls", "values"), BLOCKS)
    def test_from_dict_warns_on_unknown_keys(self, cls, values, capsys):
        cls.from_dict({"foo": 1, "bar": 2})
        captured = capsys.readouterr()
        assert "ignored unknown keys" in captured.err
        assert "'bar'" in captured.err
        assert "'foo'" in captured.err


class TestOutputConfig:
    def test_defaults(self):
        c = OutputConfig()
        assert c.write_summary_csv is True
        assert c.write_branch_csv is False
        assert c.write_graphml is False
        assert c.write_networkx_graph is False
        assert c.write_skeleton_mask is False

    def test_round_trip_dict(self):
        original = OutputConfig(
            write_summary_csv=False,
            write_branch_csv=True,
            write_graphml=True,
            write_networkx_graph=True,
            write_skeleton_mask=True,
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
        c = OutputConfig.from_dict({"write_summary_csv": 0, "write_graphml": 1})
        assert c.write_summary_csv is False
        assert c.write_graphml is True

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
            spacing=(2.0, 1.0),
            preprocessing=PreprocessingConfig(closing_iterations=1),
            postprocessing=PostprocessingConfig(junction_cleanup=True),
            feature_extraction=FeatureConfig(branches=True, fractal_dimension=True),
            output=OutputConfig(write_branch_csv=True),
        )
        restored = PipelineConfig.from_dict(original.to_dict())
        assert restored == original

    def test_defaults(self):
        c = PipelineConfig()
        assert c.schema_version == CONFIG_SCHEMA_VERSION == 8
        assert c.spacing is None
        assert c.preprocessing == PreprocessingConfig()
        assert c.output == OutputConfig()

    def test_to_dict_structure(self):
        d = PipelineConfig().to_dict()
        assert list(d) == [
            "schema_version",
            "spacing",
            "preprocessing",
            "postprocessing",
            "feature_extraction",
            "output",
        ]
        assert d["spacing"] is None

    def test_spacing_round_trips_as_tuple(self):
        d = PipelineConfig(spacing=(2.0, 1.0)).to_dict()
        assert d["spacing"] == [2.0, 1.0]
        restored = PipelineConfig.from_dict(d)
        assert restored.spacing == (2.0, 1.0)
        assert isinstance(restored.spacing, tuple)

    def test_from_dict_empty_is_defaults(self):
        assert PipelineConfig.from_dict({}) == PipelineConfig()

    def test_from_dict_unsupported_schema_version(self):
        with pytest.raises(ValueError, match="Unsupported schema_version=999"):
            PipelineConfig.from_dict({"schema_version": 999})

    def test_from_dict_rejects_schema_7_with_the_new_layout(self):
        data = {"schema_version": 7, "extraction": {}, "output": {}}
        with pytest.raises(ValueError, match="'preprocessing'.*maskel init"):
            PipelineConfig.from_dict(data)

    def test_from_dict_rejects_none(self):
        with pytest.raises(TypeError, match="must be an object"):
            PipelineConfig.from_dict(None)

    def test_from_dict_rejects_non_dict(self):
        with pytest.raises(TypeError, match="must be an object"):
            PipelineConfig.from_dict(["not", "a", "dict"])

    @pytest.mark.parametrize(
        "block", ["preprocessing", "postprocessing", "feature_extraction", "output"]
    )
    def test_from_dict_rejects_non_dict_block(self, block):
        with pytest.raises(TypeError, match=f"'{block}' must be an object"):
            PipelineConfig.from_dict({block: "bad"})

    @pytest.mark.parametrize(
        ("output_flag", "feature_flag"),
        [
            ("write_branch_csv", "branches"),
            ("write_node_csv", "nodes"),
        ],
    )
    def test_from_dict_rejects_output_flag_without_matching_feature_flag(
        self, output_flag, feature_flag
    ):
        data = {"output": {output_flag: True}}
        with pytest.raises(
            ValueError,
            match=f"output.{output_flag}=true requires "
            f"feature_extraction.{feature_flag}=true",
        ):
            PipelineConfig.from_dict(data)

    def test_from_dict_reports_every_mismatched_pair_at_once(self):
        data = {"output": {"write_branch_csv": True, "write_node_csv": True}}
        with pytest.raises(ValueError) as excinfo:
            PipelineConfig.from_dict(data)
        message = str(excinfo.value)
        assert "write_branch_csv" in message
        assert "write_node_csv" in message

    def test_from_dict_accepts_output_flag_with_matching_feature_flag(self):
        data = {
            "feature_extraction": {"branches": True, "nodes": True},
            "output": {"write_branch_csv": True, "write_node_csv": True},
        }
        c = PipelineConfig.from_dict(data)  # must not raise
        assert c.output.write_branch_csv is True

    def test_from_dict_warns_on_unknown_keys(self, capsys):
        PipelineConfig.from_dict({"extraction": {}})
        captured = capsys.readouterr()
        assert "ignored unknown keys" in captured.err
        assert "'extraction'" in captured.err


class TestConfigFileIO:
    def test_save_and_load_round_trip(self, tmp_path):
        config = PipelineConfig(
            spacing=(1.0, 0.5, 0.5),
            feature_extraction=FeatureConfig(radius_features=True),
            output=OutputConfig(write_graphml=True),
        )
        path = tmp_path / "config.json"
        save_pipeline_config(config, path)
        assert load_pipeline_config(path) == config

    def test_save_creates_valid_json(self, tmp_path):
        path = tmp_path / "pipeline.json"
        save_pipeline_config(PipelineConfig(), path)

        with path.open(encoding="utf-8") as f:
            raw = json.load(f)

        assert raw["schema_version"] == CONFIG_SCHEMA_VERSION
        assert raw["feature_extraction"]["branches"] is False
        assert raw["output"]["write_summary_csv"] is True

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline_config(tmp_path / "nonexistent.json")

    def test_load_malformed_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            load_pipeline_config(path)

    def test_save_and_load_string_paths(self, tmp_path):
        path = str(tmp_path / "str_save.json")
        save_pipeline_config(PipelineConfig(), path)
        assert load_pipeline_config(path) == PipelineConfig()
