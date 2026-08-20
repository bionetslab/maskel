"""Shared configuration models and helpers."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

CONFIG_SCHEMA_VERSION = 5

COLORABLE_BRANCH_PROPERTIES = [
    "object_id",
    "tortuosity",
    "branch-distance",
    "euclidean-distance",
    "straightness",
    "mean-pixel-value",
    "stdev-pixel-value",
    "mean_radius",
    "std_radius",
    "min_radius",
    "max_radius",
    "mean_diameter",
    "std_diameter",
    "min_diameter",
    "max_diameter",
    "volume",
    "surface_area",
]


def _warn_unknown_keys(known: set[str], data: dict[str, Any]) -> None:
    unknown = set(data) - known
    if unknown:
        print(
            f"Warning: ignored unknown keys: {sorted(unknown)}",
            file=sys.stderr,
        )


@dataclass
class ExtractionConfig:
    """Configuration for what to extract from a skeleton."""

    branches: bool = False
    branch_color_property: str = "tortuosity"
    branch_text: bool = False
    nodes: bool = False
    summary: bool = True
    fractal_dimension: bool = False
    mask_radius: bool = False
    junction_cleanup: bool = False
    cleanup_threshold_factor: float = 2.5
    prune_spurs: bool = False
    min_spur_length: float = 10.0
    spur_iterations: int = 1
    closing_iterations: int = 0
    fill_holes: bool = False
    max_hole_size: int = 0
    show_preprocessed: bool = False
    spacing: tuple[float, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "branches": self.branches,
            "branch_color_property": self.branch_color_property,
            "branch_text": self.branch_text,
            "nodes": self.nodes,
            "summary": self.summary,
            "fractal_dimension": self.fractal_dimension,
            "mask_radius": self.mask_radius,
            "junction_cleanup": self.junction_cleanup,
            "cleanup_threshold_factor": self.cleanup_threshold_factor,
            "prune_spurs": self.prune_spurs,
            "min_spur_length": self.min_spur_length,
            "spur_iterations": self.spur_iterations,
            "closing_iterations": self.closing_iterations,
            "fill_holes": self.fill_holes,
            "max_hole_size": self.max_hole_size,
            "show_preprocessed": self.show_preprocessed,
            "spacing": list(self.spacing) if self.spacing is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractionConfig:
        _warn_unknown_keys({f.name for f in fields(cls)}, data)
        kwargs = {f.name: data.get(f.name, f.default) for f in fields(cls)}
        spacing = kwargs.get("spacing")
        kwargs["spacing"] = tuple(spacing) if spacing is not None else None
        return cls(**kwargs)


@dataclass
class OutputConfig:
    """Output controls for batch CLI runs."""

    write_skeleton_npy: bool = True
    write_skeleton_png: bool = False
    write_summary_csv: bool = True
    write_branch_csv: bool = False
    write_node_csv: bool = False
    write_radius: bool = False
    write_graphml: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "write_skeleton_npy": self.write_skeleton_npy,
            "write_skeleton_png": self.write_skeleton_png,
            "write_summary_csv": self.write_summary_csv,
            "write_branch_csv": self.write_branch_csv,
            "write_node_csv": self.write_node_csv,
            "write_radius": self.write_radius,
            "write_graphml": self.write_graphml,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> OutputConfig:
        data = data or {}
        _warn_unknown_keys({f.name for f in fields(cls)}, data)
        return cls(**{f.name: bool(data.get(f.name, f.default)) for f in fields(cls)})


@dataclass
class PipelineConfig:
    """Top-level config shared by napari and batch CLI."""

    extraction: ExtractionConfig
    output: OutputConfig
    schema_version: int = CONFIG_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "extraction": self.extraction.to_dict(),
            "output": self.output.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        if not isinstance(data, dict):
            raise TypeError("Config JSON must be an object")

        _warn_unknown_keys({"schema_version", "extraction", "output"}, data)

        schema_version = int(data.get("schema_version", CONFIG_SCHEMA_VERSION))
        if schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version={schema_version}. "
                f"Expected {CONFIG_SCHEMA_VERSION}."
            )

        extraction_data = data.get("extraction", {})
        output_data = data.get("output", {})

        if not isinstance(extraction_data, dict):
            raise TypeError("'extraction' must be an object")
        if not isinstance(output_data, dict):
            raise TypeError("'output' must be an object")

        return cls(
            extraction=ExtractionConfig.from_dict(extraction_data),
            output=OutputConfig.from_dict(output_data),
            schema_version=schema_version,
        )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and parse a pipeline config from JSON file."""
    with Path(path).open() as f:
        return PipelineConfig.from_dict(json.load(f))


def save_pipeline_config(config: PipelineConfig, path: str | Path) -> None:
    """Save a pipeline config to JSON file."""
    with Path(path).open("w") as f:
        json.dump(config.to_dict(), f, indent=2)
