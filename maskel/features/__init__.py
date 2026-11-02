"""Stage 3: features of every label's skeleton graph."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from pandas import DataFrame
from skan import Skeleton, summarize

from maskel.config import ExtractionConfig
from maskel.features.fractal import fractal_dimension
from maskel.features.nodes import extract_node_features
from maskel.features.radii import (
    aggregate_segment_stats,
    per_segment_radii,
    radius_stats,
)
from maskel.features.summary import (
    _EMPTY_FEATURES,
    compute_tortuosity,
    extract_summary_features,
)
from maskel.graph import build_graph
from maskel.io.voxels import LabelVoxels, VoxelList

__all__ = [
    "_EMPTY_FEATURES",
    "LabelFeatures",
    "analyze",
    "compute_tortuosity",
    "extract_node_features",
    "extract_summary_features",
    "fractal_dimension",
]


@dataclass
class LabelFeatures:
    """Graph, branch table and feature records of one label."""

    object_id: int
    graph: Skeleton
    branch_data: DataFrame
    radius: np.ndarray | None
    summary: dict[str, float]
    branch_records: list[dict[str, object]]
    node_records: list[dict[str, object]]


def analyze(voxels: VoxelList, extraction: ExtractionConfig) -> Iterator[LabelFeatures]:
    """Features of every label of *voxels* that has a graph."""
    spacing = voxels.spacing
    include_fractal = extraction.fractal_dimension
    if include_fractal and spacing is not None and len(set(spacing)) > 1:
        print(
            "Warning: fractal_dimension is invalid for anisotropic "
            f"spacing {spacing}; forcing it to 0.0.",
            file=sys.stderr,
        )
        include_fractal = False

    for part in voxels:
        graph = build_graph(part.coords, voxels.shape, spacing)
        if graph is not None:
            yield _analyze_label(part, graph, spacing, extraction, include_fractal)


def _analyze_label(
    part: LabelVoxels,
    graph: Skeleton,
    spacing: tuple[float, ...] | None,
    extraction: ExtractionConfig,
    include_fractal: bool,
) -> LabelFeatures:
    """Features of one label's graph."""
    radius = part.radius if extraction.mask_radius else None
    branch_data = summarize(graph, separator="-")

    stats = None
    if radius is not None:
        stats = radius_stats(radius)
        if not branch_data.empty:
            for key, values in per_segment_radii(graph, radius, spacing).items():
                branch_data[key] = values
            stats.update(aggregate_segment_stats(branch_data))

    if not branch_data.empty:
        euclidean = branch_data["euclidean-distance"].to_numpy(dtype=float)
        branch_dist = branch_data["branch-distance"].to_numpy(dtype=float)
        branch_data["tortuosity"] = compute_tortuosity(branch_dist, euclidean)
        straightness = np.full_like(branch_dist, np.nan)
        valid = branch_dist > 0
        straightness[valid] = euclidean[valid] / branch_dist[valid]
        branch_data["straightness"] = straightness

    summary = {}
    if extraction.summary:
        summary = extract_summary_features(
            branch_data,
            mask_voxels=part.mask_voxels,
            crop_voxels=part.crop_voxels,
            fractal=fractal_dimension(part.coords) if include_fractal else (0.0, 0.0),
            radius_stats=stats,
            spacing=spacing,
        )
        summary["object_id"] = part.label

    branch_records = []
    if extraction.branches:
        branch_records = [
            {**row, "object_id": part.label}
            for row in branch_data.to_dict(orient="records")
        ]

    node_records = []
    if extraction.nodes:
        node_records = [
            {**row, "object_id": part.label}
            for row in extract_node_features(graph, branch_data, radius=radius)
        ]

    return LabelFeatures(
        part.label, graph, branch_data, radius, summary, branch_records, node_records
    )
