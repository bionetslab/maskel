import numpy as np
from numba import njit, prange
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skan import Skeleton

from ._utils import to_binary


@njit(parallel=True, cache=True)
def _box_count_2d(img: np.ndarray, scale: int) -> int:
    """Count occupied boxes of size `scale` in a 2D binary image."""
    H, W = img.shape
    nh = (H + scale - 1) // scale
    nw = (W + scale - 1) // scale
    count = 0
    for i in prange(nh):
        for j in range(nw):
            r0 = i * scale
            c0 = j * scale
            r1 = min(r0 + scale, H)
            c1 = min(c0 + scale, W)
            found = False
            for r in range(r0, r1):
                if found:
                    break
                for c in range(c0, c1):
                    if img[r, c]:
                        found = True
                        break
            if found:
                count += 1
    return count


@njit(parallel=True, cache=True)
def _box_count_3d(img: np.ndarray, scale: int) -> int:
    """Count occupied boxes of size `scale` in a 3D binary volume."""
    D, H, W = img.shape
    nd = (D + scale - 1) // scale
    nh = (H + scale - 1) // scale
    nw = (W + scale - 1) // scale
    count = 0
    for i in prange(nd):
        for j in range(nh):
            for k in range(nw):
                z0 = i * scale
                y0 = j * scale
                x0 = k * scale
                z1 = min(z0 + scale, D)
                y1 = min(y0 + scale, H)
                x1 = min(x0 + scale, W)
                found = False
                for z in range(z0, z1):
                    if found:
                        break
                    for y in range(y0, y1):
                        if found:
                            break
                        for x in range(x0, x1):
                            if img[z, y, x]:
                                found = True
                                break
                if found:
                    count += 1
    return count


def compute_tortuosity(lengths: np.ndarray, euclidean: np.ndarray) -> np.ndarray:
    """Compute tortuosity as branch-distance / euclidean-distance.

    Returns NaN where euclidean distance is zero.
    """
    tortuosity = np.full_like(lengths, np.nan, dtype=float)
    valid = euclidean > 0
    tortuosity[valid] = lengths[valid] / euclidean[valid]
    return tortuosity


def fractal_dimension(skeleton: np.ndarray) -> tuple[float, float]:
    """Estimate fractal dimension of a binary skeleton via box-counting.

    Returns
    -------
    fd : float
        Fractal dimension (slope magnitude of log-log fit).
    r2 : float
        R^2 of the log-log linear fit (fit quality indicator).
    """
    binary = to_binary(skeleton)
    min_side = min(binary.shape)
    scale_base = min_side // 4
    if scale_base < 1:
        return 0.0, 0.0
    max_exp = int(np.floor(np.log2(scale_base)))
    if max_exp < 1:
        return 0.0, 0.0

    scales = np.array([2**k for k in range(1, max_exp + 1)], dtype=np.int64)

    box_counter = _box_count_2d if binary.ndim == 2 else _box_count_3d
    counts = np.array([box_counter(binary, int(s)) for s in scales], dtype=np.float64)

    # scales with zero count collapses the log
    valid = counts > 0
    if valid.sum() < 2:
        return 0.0, 0.0

    log_s = np.log(scales[valid].astype(np.float64))
    log_n = np.log(counts[valid])

    coeffs = np.polyfit(log_s, log_n, 1)
    fd = float(-coeffs[0])

    predicted = np.polyval(coeffs, log_s)
    ss_res = float(np.sum((log_n - predicted) ** 2))
    ss_tot = float(np.sum((log_n - log_n.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return fd, float(r2)


def compute_radii(
    binary: np.ndarray,
    skeleton: np.ndarray,
    spacing: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute vessel radii via Euclidean distance transform of the binary mask.

    The distance transform gives each foreground pixel its distance to the
    nearest background pixel. Sampling these values at skeleton positions
    yields the local vessel radius at every centerline point.

    Parameters
    ----------
    binary : ndarray
        Binary vessel mask (foreground > 0).
    skeleton : ndarray
        Binary skeleton of the same shape.
    spacing : tuple[float, ...], optional
        Per-axis physical size of one pixel/voxel. ``None`` (the default)
        keeps isotropic unit spacing, matching scipy's own default.

    Returns
    -------
    radius_matrix : ndarray, shape=binary.shape, dtype=float64
        Array with radius values at skeleton pixels and zero elsewhere.
    stats : dict[str, float]
        Aggregated statistics (keys: ``mean_radius``, ``std_radius``,
        ``min_radius``, ``max_radius``, ``mean_diameter``, ``std_diameter``,
        ``min_diameter``, ``max_diameter``).
    """
    edt = distance_transform_edt(binary, sampling=spacing)
    radius_matrix = np.zeros_like(binary, dtype=np.float64)
    radius_matrix[skeleton > 0] = edt[skeleton > 0]

    radii = radius_matrix[radius_matrix > 0]
    if radii.size:
        mean_r = float(np.mean(radii))
        std_r = float(np.std(radii))
        min_r = float(np.min(radii))
        max_r = float(np.max(radii))
    else:
        mean_r = std_r = min_r = max_r = 0.0

    return radius_matrix, {
        "mean_radius": mean_r,
        "std_radius": std_r,
        "min_radius": min_r,
        "max_radius": max_r,
        "mean_diameter": 2.0 * mean_r,
        "std_diameter": 2.0 * std_r,
        "min_diameter": 2.0 * min_r,
        "max_diameter": 2.0 * max_r,
    }


def per_segment_radii(
    radius_matrix: np.ndarray,
    graph: Skeleton,
    n_branches: int,
    spacing: tuple[float, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-segment radius/diameter statistics from an EDT radius matrix.

    Parameters
    ----------
    radius_matrix : ndarray
        Array with radius values at skeleton pixels, zero elsewhere.
    graph : Skeleton
        Pre-built skan Skeleton graph.
    n_branches : int
        Number of branches (e.g. ``len(branch_data)``).
    spacing : tuple[float, ...], optional
        Per-axis physical size of one pixel/voxel, used to weight each
        step along a branch's path by its real physical length rather
        than one unit per pixel. ``None`` (the default) is equivalent to
        isotropic unit spacing.

    Returns
    -------
    dict[str, ndarray]
        Arrays of length ``n_branches`` with keys ``mean_radius``,
        ``std_radius``, ``min_radius``, ``max_radius``, ``mean_diameter``,
        ``std_diameter``, ``min_diameter``, ``max_diameter``, ``volume``,
        ``surface_area``.
        Branches with no positive radius values yield ``nan``.

    Notes
    -----
    ``volume``/``surface_area`` model the branch as a chain of frustums
    between consecutive skeleton pixels along its path: each step's
    physical length (accounting for diagonal steps, and for anisotropic
    *spacing*) is weighted by the average of its two endpoints' radii,
    rather than treating every pixel as contributing one unit of length
    regardless of step direction.
    """
    n = n_branches
    mean_r = np.full(n, np.nan, dtype=np.float64)
    std_r = np.full(n, np.nan, dtype=np.float64)
    min_r = np.full(n, np.nan, dtype=np.float64)
    max_r = np.full(n, np.nan, dtype=np.float64)
    volume = np.full(n, np.nan, dtype=np.float64)
    surface_area = np.full(n, np.nan, dtype=np.float64)

    spacing_arr = np.asarray(spacing, dtype=np.float64) if spacing is not None else None

    for i in range(n):
        coords = graph.path_coordinates(i)
        radii = radius_matrix[tuple(coords.T)]
        radii_valid = radii[radii > 0]
        if radii_valid.size:
            mean_r[i] = np.mean(radii_valid)
            std_r[i] = np.std(radii_valid)
            min_r[i] = np.min(radii_valid)
            max_r[i] = np.max(radii_valid)

            # Model the branch as a chain of frustums between consecutive
            # path points: each step's physical length (diagonal-aware,
            # and spacing-aware for anisotropic voxels) is weighted by the
            # average of its two endpoints' radii.
            steps = np.diff(coords, axis=0).astype(np.float64)
            if spacing_arr is not None:
                steps = steps * spacing_arr
            step_lengths = np.linalg.norm(steps, axis=1)
            r_mid = (radii[:-1] + radii[1:]) / 2.0

            volume[i] = np.pi * float(np.sum(r_mid**2 * step_lengths))
            surface_area[i] = 2.0 * np.pi * float(np.sum(r_mid * step_lengths))

    return {
        "mean_radius": mean_r,
        "std_radius": std_r,
        "min_radius": min_r,
        "max_radius": max_r,
        "mean_diameter": 2.0 * mean_r,
        "std_diameter": 2.0 * std_r,
        "min_diameter": 2.0 * min_r,
        "max_diameter": 2.0 * max_r,
        "volume": volume,
        "surface_area": surface_area,
    }


def build_skeleton_graph(
    skeleton: np.ndarray, spacing: tuple[float, ...] | None = None
) -> Skeleton:
    """Build a graph representation from a binary vessel skeleton.

    Parameters
    ----------
    skeleton : ndarray
        Binary skeleton array.
    spacing : tuple[float, ...], optional
        Per-axis physical size of one pixel/voxel, passed through to
        ``skan.Skeleton``. Once built with spacing, everything downstream
        via ``skan.summarize`` (``branch-distance``, ``euclidean-distance``,
        and therefore ``total_length``/``mean_length``/``tortuosity``/``hgu``)
        becomes physically correct automatically. ``graph.coordinates``
        stay raw pixel indices regardless of spacing. ``None`` (the
        default) keeps isotropic unit spacing.
    """
    return Skeleton(to_binary(skeleton), spacing=spacing if spacing is not None else 1)


def extract_node_features(
    graph: Skeleton,
    branch_data,
    *,
    radius_matrix: np.ndarray | None = None,
) -> list[dict[str, object]]:
    """Extract per-node features from the skeleton graph.

    Returns one dict per graph node, with keys:
    ``node_id``, ``coord_0`` … ``coord_{ndim-1}``, ``degree``,
    ``is_endpoint``, ``is_junction``, ``is_pass_through``,
    and ``radius`` (if *radius_matrix* provided).
    """
    n_nodes = graph.coordinates.shape[0]
    ndim = graph.coordinates.shape[1]

    records: list[dict[str, object]] = []
    for node_id in range(n_nodes):
        degree = int(graph.degrees[node_id])
        coords = graph.coordinates[node_id]

        record: dict[str, object] = {
            "node_id": node_id,
            "degree": degree,
            "is_endpoint": degree == 1,
            "is_junction": degree >= 3,
            "is_pass_through": degree == 2,
        }

        for d in range(ndim):
            record[f"coord_{d}"] = int(coords[d])

        if radius_matrix is not None:
            record["radius"] = float(radius_matrix[tuple(coords)])

        records.append(record)

    return records


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
    "vessel_area": 0.0,
    "vessel_area_fraction": 0.0,
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


def aggregate_segment_stats(branch_data) -> dict[str, float]:
    """Object-level aggregates of the per-branch ``volume``/``surface_area``
    columns produced by `per_segment_radii` (mean across this object's
    branches).

    Requires *branch_data* to already carry those two columns (only true
    after `per_segment_radii`'s output has been merged into it) and to be
    non-empty - a caller that hasn't done that, or has no branches, should
    not call this.
    """
    return {
        "mean_segment_volume": float(np.nanmean(branch_data["volume"])),
        "mean_surface_area": float(np.nanmean(branch_data["surface_area"])),
    }


def extract_summary_features(
    skeleton: np.ndarray,
    graph: Skeleton,
    branch_data,
    *,
    binary: np.ndarray,
    include_fractal: bool = True,
    radius_stats: dict[str, float] | None = None,
    spacing: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Extract graph-topology and segment statistics from a vessel skeleton.

    Parameters
    ----------
    skeleton : ndarray
        Binary 2D or 3D skeleton array. Used only for fractal dimension
        computation, not for graph topology.
    graph : Skeleton
        Pre-built skan Skeleton graph (e.g. from `build_skeleton_graph`).
    branch_data : DataFrame
        Pre-computed branch summary (e.g. from `skan.summarize(graph, ...)`).
    binary : ndarray
        Original binary mask. Used to compute ``vessel_area`` and
        ``vessel_area_fraction``.
    include_fractal : bool, optional
        Whether to compute fractal dimension (expensive-ish). Default is True.
    radius_stats : dict[str, float], optional
        Pre-computed radius/diameter statistics from `compute_radii`
        (``mean_radius``, ``std_radius``, ``min_radius``, ``max_radius``,
        ``mean_diameter``, ``std_diameter``, ``min_diameter``,
        ``max_diameter``). Note ``compute_radii`` alone does *not* produce
        ``mean_segment_volume``/``mean_surface_area`` - those are
        branch-level aggregates from `aggregate_segment_stats`, which needs
        `per_segment_radii`'s output merged into *branch_data* first. Any
        key missing from *radius_stats* (including both of those, if you
        only ran `compute_radii`) defaults to ``0.0`` rather than being
        silently dropped from the returned dict - the schema is always the
        same 10 keys regardless of what *radius_stats* actually supplies.
        ``None`` defaults every one of them to ``0.0``.
    spacing : tuple[float, ...], optional
        Per-axis physical size of one pixel/voxel. When given, scales
        ``vessel_area`` from a raw pixel/voxel count into physical units
        (``vessel_area_fraction`` needs no change since it's a ratio and
        the scaling cancels). ``None`` (the default) keeps pixel units.
    """
    if branch_data.empty:
        return dict(_EMPTY_FEATURES)

    vessel_area = float(np.count_nonzero(binary))
    if spacing is not None:
        vessel_area *= float(np.prod(spacing))
    vessel_area_fraction = float(np.count_nonzero(binary)) / float(binary.size)

    fd, fd_r2 = fractal_dimension(skeleton) if include_fractal else (0.0, 0.0)

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
        "vessel_area": vessel_area,
        "vessel_area_fraction": vessel_area_fraction,
        **radius,
    }
