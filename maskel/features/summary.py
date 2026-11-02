"""Per-object summary features: topology, branch lengths, tortuosity."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def compute_tortuosity(lengths: np.ndarray, euclidean: np.ndarray) -> np.ndarray:
    """Compute tortuosity as branch-distance / euclidean-distance.

    Returns NaN where euclidean distance is zero.
    """
    tortuosity = np.full_like(lengths, np.nan, dtype=float)
    valid = euclidean > 0
    tortuosity[valid] = lengths[valid] / euclidean[valid]
    return tortuosity


_EMPTY_FEATURES: dict[str, float] = {
    "num_nodes": 0.0,
    "num_edges": 0.0,
    "num_endpoints": 0.0,
    "num_bifurcations": 0.0,
    "total_length": 0.0,
    "mean_length": 0.0,
    "std_length": 0.0,
    "max_length": 0.0,
    "min_length": 0.0,
    "mean_tortuosity": 0.0,
    "std_tortuosity": 0.0,
    "num_components": 0.0,
    "mean_degree": 0.0,
    "max_degree": 0.0,
    "fractal_dimension": 0.0,
    "fractal_dimension_r2": 0.0,
    "hgu": 0.0,
    "mean_radius": 0.0,
    "std_radius": 0.0,
    "min_radius": 0.0,
    "max_radius": 0.0,
    "mean_diameter": 0.0,
    "std_diameter": 0.0,
    "min_diameter": 0.0,
    "max_diameter": 0.0,
    "mask_area": 0.0,
    "mask_area_fraction": 0.0,
    "mean_segment_volume": 0.0,
    "mean_surface_area": 0.0,
}


_RADIUS_STATS_DEFAULTS: dict[str, float] = {
    "mean_radius": 0.0,
    "std_radius": 0.0,
    "min_radius": 0.0,
    "max_radius": 0.0,
    "mean_diameter": 0.0,
    "std_diameter": 0.0,
    "min_diameter": 0.0,
    "max_diameter": 0.0,
    "mean_segment_volume": 0.0,
    "mean_surface_area": 0.0,
}


def extract_summary_features(
    branch_data,
    *,
    mask_voxels: int,
    crop_voxels: int,
    fractal: tuple[float, float] = (0.0, 0.0),
    radius_stats: dict[str, float] | None = None,
    spacing: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Topology, length, tortuosity, mask and radius statistics of one label's branch table."""
    if branch_data.empty:
        return dict(_EMPTY_FEATURES)

    mask_area = float(mask_voxels)
    if spacing is not None:
        mask_area *= float(np.prod(spacing))
    mask_area_fraction = float(mask_voxels) / float(crop_voxels)
    fd, fd_r2 = fractal

    src_nodes = branch_data["node-id-src"].to_numpy(dtype=np.int64)
    dst_nodes = branch_data["node-id-dst"].to_numpy(dtype=np.int64)
    edge_nodes = np.concatenate((src_nodes, dst_nodes))
    unique_nodes = np.unique(edge_nodes)

    num_edges = len(branch_data)
    num_nodes = int(unique_nodes.size)

    max_node_id = int(np.max(unique_nodes))
    degrees = np.bincount(edge_nodes, minlength=max_node_id + 1)
    node_degrees = degrees[unique_nodes]

    num_endpoints = int(np.count_nonzero(node_degrees == 1))
    num_bifurcations = int(np.count_nonzero(node_degrees >= 3))
    mean_degree = float(np.mean(node_degrees)) if node_degrees.size else 0.0
    max_degree = float(np.max(node_degrees)) if node_degrees.size else 0.0

    node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
    src_idx = np.fromiter((node_to_index[src] for src in src_nodes), dtype=np.int64)
    dst_idx = np.fromiter((node_to_index[dst] for dst in dst_nodes), dtype=np.int64)
    adjacency = coo_matrix(
        (
            np.ones(src_idx.size * 2, dtype=np.uint8),
            (
                np.concatenate((src_idx, dst_idx)),
                np.concatenate((dst_idx, src_idx)),
            ),
        ),
        shape=(num_nodes, num_nodes),
    ).tocsr()
    num_components = int(
        connected_components(adjacency, directed=False, return_labels=False)
    )

    lengths = branch_data["branch-distance"].to_numpy(dtype=float)
    euclidean = branch_data["euclidean-distance"].to_numpy(dtype=float)

    if lengths.size:
        total_length = float(np.sum(lengths))
        mean_length = float(np.mean(lengths))
        std_length = float(np.std(lengths))
        max_length = float(np.max(lengths))
        min_length = float(np.min(lengths))
    else:
        total_length = 0.0
        mean_length = 0.0
        std_length = 0.0
        max_length = 0.0
        min_length = 0.0

    tortuosity = compute_tortuosity(lengths, euclidean)
    tortuosity = tortuosity[~np.isnan(tortuosity)]
    if tortuosity.size:
        mean_tortuosity = float(np.mean(tortuosity))
        std_tortuosity = float(np.std(tortuosity))
    else:
        mean_tortuosity = 0.0
        std_tortuosity = 0.0

    hgu = total_length / float(num_endpoints) if num_endpoints else 0.0

    radius_stats = radius_stats or {}
    radius = {
        key: radius_stats.get(key, default)
        for key, default in _RADIUS_STATS_DEFAULTS.items()
    }

    return {
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
        "num_endpoints": float(num_endpoints),
        "num_bifurcations": float(num_bifurcations),
        "total_length": total_length,
        "mean_length": mean_length,
        "std_length": std_length,
        "max_length": max_length,
        "min_length": min_length,
        "mean_tortuosity": mean_tortuosity,
        "std_tortuosity": std_tortuosity,
        "num_components": float(num_components),
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "fractal_dimension": fd,
        "fractal_dimension_r2": fd_r2,
        "hgu": hgu,
        "mask_area": mask_area,
        "mask_area_fraction": mask_area_fraction,
        **radius,
    }
