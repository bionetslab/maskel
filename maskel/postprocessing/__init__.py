"""Stage 2: skeleton cleanup on the graph - junction collapsing and spur pruning."""

from __future__ import annotations

import numpy as np

from maskel.config import ExtractionConfig
from maskel.graph import build_graph
from maskel.io.voxels import LabelVoxels, VoxelList
from maskel.postprocessing.junction_cleanup import collapse_triangle_junctions
from maskel.postprocessing.spur_pruning import prune_short_spurs

__all__ = ["collapse_triangle_junctions", "postprocess", "prune_short_spurs"]


def postprocess(voxels: VoxelList, extraction: ExtractionConfig) -> VoxelList:
    """Apply the configured cleanup to every label of *voxels*."""
    return voxels.with_labels(
        [_postprocess_label(part, voxels, extraction) for part in voxels]
    )


def _postprocess_label(
    part: LabelVoxels, voxels: VoxelList, extraction: ExtractionConfig
) -> LabelVoxels:
    """Cleaned-up voxels of one label."""
    coords, radius = part.coords, part.radius
    graph = build_graph(coords, voxels.shape, voxels.spacing)
    if graph is None:
        return part

    if extraction.junction_cleanup:
        coords, radius = collapse_triangle_junctions(
            graph, radius, extraction.cleanup_threshold_factor
        )
        graph = build_graph(coords, voxels.shape, voxels.spacing)
        if graph is None:
            # nothing connected is left after collapsing
            coords, radius = coords[:0], radius[:0]

    if extraction.prune_spurs:
        # pruning can turn a junction into a new spur tip, hence the repeats
        for _ in range(max(1, extraction.spur_iterations)):
            if graph is None:
                break
            keep = prune_short_spurs(graph, extraction.min_spur_length)
            coords, radius = coords[keep], radius[keep]
            graph = build_graph(coords, voxels.shape, voxels.spacing)

    return LabelVoxels(
        part.label,
        np.asarray(coords, dtype=np.int64),
        radius,
        part.mask_voxels,
        part.crop_voxels,
    )
