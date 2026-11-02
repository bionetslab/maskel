"""Removal of short spur branches (thinning artifacts) from a skeleton graph."""

import numpy as np
from skan import summarize
from skan.csr import Skeleton


def prune_short_spurs(graph: Skeleton, min_length: float = 10.0) -> np.ndarray:
    """Mask over *graph*'s voxels that drops every endpoint-to-junction branch shorter than *min_length*."""
    keep = np.ones(len(graph.coordinates), dtype=bool)
    branch_data = summarize(graph, separator="-")
    if branch_data.empty:
        return keep

    src = branch_data["node-id-src"].to_numpy(dtype=np.int64)
    dst = branch_data["node-id-dst"].to_numpy(dtype=np.int64)
    src_degree = graph.degrees[src]
    dst_degree = graph.degrees[dst]
    is_spur = (
        ((src_degree == 1) & (dst_degree > 1)) | ((dst_degree == 1) & (src_degree > 1))
    ) & (branch_data["branch-distance"].to_numpy(dtype=float) < min_length)

    if is_spur.all():
        # every branch got pruned away - nothing left worth keeping
        keep[:] = False
        return keep
    for i in np.flatnonzero(is_spur):
        # path[0] is the src node, path[-1] the dst node; keep the junction
        # voxel so the rest of the skeleton stays connected
        path = graph.path(i)
        keep[path[:-1] if src_degree[i] == 1 else path[1:]] = False
    return keep
