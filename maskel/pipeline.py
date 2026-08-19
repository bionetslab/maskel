"""Shared analysis pipeline used by the CLI (and by napari-maskel, which wraps it)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage as ndi
from skan import summarize

from maskel._utils import to_binary
from maskel.config import PipelineConfig
from maskel.features import (
    build_vessel_graph,
    compute_radii,
    compute_tortuosity,
    extract_node_features,
    extract_vessel_features,
    per_segment_radii,
)
from maskel.junction_cleanup import collapse_triangle_junctions
from maskel.spur_pruning import prune_short_spurs
from maskel.thin import lee94_thin

if TYPE_CHECKING:
    from pandas import DataFrame
    from skan import Skeleton


@dataclass
class AnalysisResult:
    """Container for single-image analysis outputs.

    Deliberately napari-free: napari-maskel builds its own LayerDataTuples
    from these fields rather than this package returning them directly.
    """

    skeleton: np.ndarray
    summary_features: dict[str, float]
    branch_records: list[dict[str, object]]
    node_records: list[dict[str, object]]
    radius_matrix: np.ndarray | None = None
    preprocessed_binary: np.ndarray | None = None
    graph: Skeleton | None = None
    branch_data: DataFrame | None = None


def preprocess_binary(
    binary: np.ndarray,
    closing_iterations: int = 0,
    fill_holes: bool = False,
    max_hole_size: int = 0,
) -> np.ndarray:
    """Apply morphological preprocessing to a binary mask.

    Parameters
    ----------
    binary : ndarray
        Binary uint8 array (0/1).
    closing_iterations : int
        Number of binary closing iterations (0 = skip).
    fill_holes : bool
        Whether to fill enclosed background regions.
    max_hole_size : int
        Max hole size in voxels to fill (0 = unlimited).

    Returns
    -------
    ndarray
        Preprocessed binary array.
    """
    if closing_iterations > 0:
        structure = ndi.generate_binary_structure(binary.ndim, 1)
        binary = ndi.binary_closing(
            binary, structure=structure, iterations=closing_iterations
        ).astype(binary.dtype)

    if fill_holes:
        before_fill = binary.copy() if max_hole_size > 0 else None
        binary = ndi.binary_fill_holes(binary).astype(binary.dtype)

        if max_hole_size > 0 and before_fill is not None:
            diff = binary.astype(np.int8) - before_fill.astype(np.int8)
            filled = diff > 0
            if filled.any():
                labels, _ = ndi.label(filled)
                sizes = np.bincount(labels.ravel())
                big = sizes > max_hole_size
                big[0] = False
                revert = big[labels]
                binary[revert] = 0

    return binary


def analyze_binary_image(
    image: np.ndarray,
    config: PipelineConfig,
) -> AnalysisResult:
    """Run the full skeletonization + extraction pipeline for one image.

    Parameters
    ----------
    image : ndarray
        Input image array. Non-zero values are treated as foreground.
    config : PipelineConfig
        Full pipeline configuration.
    """
    binary = to_binary(image)

    # -- optional: morphological preprocessing -------------------------
    preprocessed_binary: np.ndarray | None = None
    if config.extraction.closing_iterations > 0 or config.extraction.fill_holes:
        binary = preprocess_binary(
            binary,
            closing_iterations=config.extraction.closing_iterations,
            fill_holes=config.extraction.fill_holes,
            max_hole_size=config.extraction.max_hole_size,
        )
        preprocessed_binary = binary

    skeleton = lee94_thin(binary)

    if not skeleton.any():
        return AnalysisResult(
            skeleton=skeleton,
            summary_features={},
            branch_records=[],
            node_records=[],
        )

    # -- optional: collapse triangle junction artifacts -----------------
    # (requires EDT; done on the original skeleton before graph building)
    if config.extraction.junction_cleanup:
        rm_temp, _ = compute_radii(binary, skeleton)
        skeleton = collapse_triangle_junctions(
            skeleton,
            radius_matrix=rm_temp,
            threshold_factor=config.extraction.cleanup_threshold_factor,
        )

    # -- build graph & branch data on the (potentially cleaned) skeleton -
    graph = build_vessel_graph(skeleton)

    # -- optional: prune short spur branches -----------------------------
    # (a spur: one node is an endpoint (degree 1), the other a junction
    # (degree > 1), and the branch is shorter than the configured
    # threshold - usually a thinning artifact rather than a real vessel
    # tip. Pruning changes the skeleton's pixels, so the graph is rebuilt
    # from scratch afterwards.)
    if config.extraction.prune_spurs:
        # Pruning may expose new short spurs (an endpoint's parent junction
        # can become an endpoint once its own spur is removed), so the
        # operation is repeated on its own output up to spur_iterations times.
        for _ in range(max(1, config.extraction.spur_iterations)):
            pre_prune_branch_data = summarize(graph, separator="-")
            skeleton = prune_short_spurs(
                skeleton,
                graph,
                pre_prune_branch_data,
                min_length=config.extraction.min_spur_length,
            )
            if not skeleton.any():
                return AnalysisResult(
                    skeleton=skeleton,
                    summary_features={},
                    branch_records=[],
                    node_records=[],
                )
            graph = build_vessel_graph(skeleton)

    # -- optional: vessel radius (EDT on the final skeleton) ------------
    radius_matrix = None
    radius_stats = None
    if config.extraction.vessel_radius:
        radius_matrix, radius_stats = compute_radii(binary, skeleton)

    branch_data = summarize(graph, separator="-")

    if radius_matrix is not None and not branch_data.empty:
        per_seg = per_segment_radii(radius_matrix, graph, len(branch_data))
        for key, arr in per_seg.items():
            branch_data[key] = arr

        if radius_stats is not None:
            radius_stats["mean_segment_volume"] = float(
                np.nanmean(branch_data["volume"])
            )
            radius_stats["mean_surface_area"] = float(
                np.nanmean(branch_data["surface_area"])
            )

    if not branch_data.empty:
        euclidean = branch_data["euclidean-distance"].to_numpy(dtype=float)
        branch_dist = branch_data["branch-distance"].to_numpy(dtype=float)
        tortuosity = compute_tortuosity(branch_dist, euclidean)
        branch_data["tortuosity"] = tortuosity
        straightness = np.full_like(branch_dist, np.nan, dtype=float)
        valid = branch_dist > 0
        straightness[valid] = euclidean[valid] / branch_dist[valid]
        branch_data["straightness"] = straightness

    summary_features: dict[str, float] = {}
    if config.extraction.summary:
        summary_features = extract_vessel_features(
            skeleton,
            graph,
            branch_data,
            binary=binary,
            include_fractal=config.extraction.fractal_dimension,
            radius_stats=radius_stats,
        )

    branch_records: list[dict[str, object]] = []
    if config.extraction.branches and not branch_data.empty:
        branch_records = branch_data.to_dict(orient="records")

    node_records: list[dict[str, object]] = []
    if config.extraction.nodes:
        node_records = extract_node_features(
            graph,
            branch_data,
            radius_matrix=radius_matrix,
        )

    return AnalysisResult(
        skeleton=skeleton,
        summary_features=summary_features,
        branch_records=branch_records,
        node_records=node_records,
        radius_matrix=radius_matrix,
        preprocessed_binary=preprocessed_binary,
        graph=graph,
        branch_data=branch_data,
    )
