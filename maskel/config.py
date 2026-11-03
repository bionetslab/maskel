"""Shared configuration models and helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 8

_BLOCKS = ("preprocessing", "postprocessing", "feature_extraction", "output")


def _warn_unknown_keys(known: set[str], data: dict[str, Any]) -> None:
    unknown = set(data) - known
    if unknown:
        print(
            f"Warning: ignored unknown keys: {sorted(unknown)}",
            file=sys.stderr,
        )


def _from_dict(cls, data: dict[str, Any]):
    """An instance of the dataclass *cls* from *data*, defaults for missing keys."""
    _warn_unknown_keys({f.name for f in fields(cls)}, data)
    return cls(**{f.name: data.get(f.name, f.default) for f in fields(cls)})


# output.<flag> -> the feature_extraction.<flag> it requires to actually
# produce anything - without the paired feature flag, the output flag is a
# silent no-op (nothing is written, no error, no warning) rather than the
# error a misconfigured "requires X" combination should be.
_OUTPUT_REQUIRES_FEATURE = {
    "write_branch_csv": "branches",
    "write_node_csv": "nodes",
}


def _validate_output_requirements(
    features: FeatureConfig, output: OutputConfig
) -> None:
    problems = [
        f"output.{out_flag}=true requires feature_extraction.{req_flag}=true"
        for out_flag, req_flag in _OUTPUT_REQUIRES_FEATURE.items()
        if getattr(output, out_flag) and not getattr(features, req_flag)
    ]
    if problems:
        raise ValueError("Invalid config:\n  " + "\n  ".join(problems))


@dataclass
class PreprocessingConfig:
    """Morphological clean-up of the mask before thinning."""

    opening_radius: int = 0
    closing_iterations: int = 0
    fill_holes: bool = False
    max_hole_diameter: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessingConfig:
        return _from_dict(cls, data)


@dataclass
class PostprocessingConfig:
    """Graph clean-up of the skeleton after thinning."""

    junction_cleanup: bool = False
    cleanup_threshold_factor: float = 2.5
    prune_spurs: bool = False
    min_spur_length: float = 10.0
    spur_iterations: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PostprocessingConfig:
        return _from_dict(cls, data)


@dataclass
class FeatureConfig:
    """Which features to compute from the skeleton."""

    summary: bool = True
    branches: bool = False
    nodes: bool = False
    fractal_dimension: bool = False
    radius_features: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureConfig:
        return _from_dict(cls, data)


@dataclass
class OutputConfig:
    """Output controls for batch CLI runs."""

    write_summary_csv: bool = True
    write_branch_csv: bool = False
    write_node_csv: bool = False
    write_graphml: bool = False
    write_networkx_graph: bool = False
    write_skeleton_mask: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OutputConfig:
        data = data or {}
        _warn_unknown_keys({f.name for f in fields(cls)}, data)
        return cls(**{f.name: bool(data.get(f.name, f.default)) for f in fields(cls)})


@dataclass
class PipelineConfig:
    """Top-level config: the spacing plus one block per stage and the outputs."""

    spacing: tuple[float, ...] | None = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    feature_extraction: FeatureConfig = field(default_factory=FeatureConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    schema_version: int = CONFIG_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spacing": list(self.spacing) if self.spacing is not None else None,
            **{block: getattr(self, block).to_dict() for block in _BLOCKS},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        if not isinstance(data, dict):
            raise TypeError("Config JSON must be an object")

        schema_version = int(data.get("schema_version", CONFIG_SCHEMA_VERSION))
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={schema_version}. "
                f"Expected {CONFIG_SCHEMA_VERSION}, which has a top-level 'spacing' "
                f"and the blocks {', '.join(repr(b) for b in _BLOCKS)}. "
                f"Create a new config with `maskel init`."
            )

        _warn_unknown_keys({"schema_version", "spacing", *_BLOCKS}, data)

        blocks = {block: data.get(block, {}) for block in _BLOCKS}
        for block, block_data in blocks.items():
            if not isinstance(block_data, dict):
                raise TypeError(f"'{block}' must be an object")

        spacing = data.get("spacing")
        config = cls(
            spacing=tuple(spacing) if spacing is not None else None,
            preprocessing=PreprocessingConfig.from_dict(blocks["preprocessing"]),
            postprocessing=PostprocessingConfig.from_dict(blocks["postprocessing"]),
            feature_extraction=FeatureConfig.from_dict(blocks["feature_extraction"]),
            output=OutputConfig.from_dict(blocks["output"]),
            schema_version=schema_version,
        )
        _validate_output_requirements(config.feature_extraction, config.output)
        return config


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and parse a pipeline config from JSON file."""
    with Path(path).open() as f:
        return PipelineConfig.from_dict(json.load(f))


def save_pipeline_config(config: PipelineConfig, path: str | Path) -> None:
    """Save a pipeline config to JSON file."""
    with Path(path).open("w") as f:
        json.dump(config.to_dict(), f, indent=2)
