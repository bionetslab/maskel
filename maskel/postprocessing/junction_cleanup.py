import warnings
from typing import Any

import networkx as nx
import numpy as np
from skan import summarize
from skan.csr import Skeleton, skeleton_to_nx


def _bresenham_line(
    p1: tuple[int, ...],
    p2: tuple[int, ...],
) -> list[tuple[int, ...]]:
    """Integer-coordinate points along a line from *p1* to *p2*."""
    ndim = len(p1)
    delta = [p2[d] - p1[d] for d in range(ndim)]
    n_steps = max(abs(d) for d in delta)
    if n_steps == 0:
        return [p1]
    result: list[tuple[int, ...]] = []
    # walk along the line in equal fractions of the total span
    for i in range(n_steps + 1):
        # arithmetic rounding (``int(x + 0.5)``) to stay on the expected raster
        pt = tuple(int(p1[d] + i * delta[d] / n_steps + 0.5) for d in range(ndim))
        result.append(pt)
    return result


def collapse_triangle_junctions(
    graph: Skeleton,
    radius: np.ndarray,
    threshold_factor: float = 2.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxels and radii of *graph* after collapsing cycles smaller than *threshold_factor* local diameters."""

    shape = tuple(graph.skeleton_shape)

    # 1. junction graph: nodes are junctions/endpoints, edges the paths
    unchanged = graph.coordinates, radius
    summary = summarize(graph, separator="-")
    if summary.empty:
        return unchanged

    G = skeleton_to_nx(graph, summary)

    # 2. node -> pixel coordinate lookup
    # junction graph nodes are pixel ids, i.e. rows of graph.coordinates.
    # (Not from the edge paths: networkx may report an edge as (dst, src)
    # while its path still runs from src to dst.)
    node_coords: dict[int, tuple[int, ...]] = {
        n: tuple(int(c) for c in graph.coordinates[n]) for n in G.nodes
    }

    # 3. find cycles
    # skan produces a MultiGraph but nx.cycle_basis requires a plain Graph
    # cycle_basis returns one representative cycle per elementary cycle,
    # each as a list of node IDs.
    try:
        G_simple = nx.Graph(G)
        cycles = nx.cycle_basis(G_simple)
    except (nx.NetworkXException, KeyError, ValueError) as exc:
        warnings.warn(
            f"junction_cleanup: cycle detection failed ({exc!r}); "
            "skipping cleanup for this object.",
            stacklevel=2,
        )
        return unchanged

    if not cycles:
        return unchanged

    # 4. perimeter & local diameter per cycle
    # for each cycle we compute:
    #   a) perimeter - sum of Euclidean distances along the edge paths
    #      that form the cycle.
    #   b) local diameter - estimated from the radius at each
    #      cycle node, or 1 when none is positive.
    #
    # a legit branch point (like a bifurcation) will have a wide
    # perimeter relative to the local diameter. A spurious triangle-
    # or diamond-shaped junction artifact will be small and tight.
    # then use the ratio perimeter / diameter to tell them apart.
    perimeters: list[float] = []
    diameters: list[float] = []
    lookup = _RadiusLookup(graph.coordinates, radius, shape)

    for cycle in cycles:
        # walk consecutive node pairs (u, v) around the cycle.
        # The modulo wrap-around (i+1) % len(cycle) closes the loop
        # so the last node pairs back to the first.
        perim = 0.0
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            if not G.has_edge(u, v):
                perim = float("inf")  # obacht, edge must exist!
                break
            for key in G[u][v]:
                path = G[u][v][key]["path"]
                # sum euclidean dist of skeleton segments
                diffs = np.diff(path.astype(np.float64), axis=0)
                perim += float(np.sum(np.sqrt((diffs**2).sum(axis=1))))
                break  # multigraph can have multiple edges, just take first one

        perimeters.append(perim)

        # Estimate local diameter from the radius at each junction node:
        # the mean radius, doubled.
        radii = [lookup.get(node_coords[n], 0.0) for n in cycle if n in node_coords]
        radii = [r for r in radii if r > 0]
        diameters.append(2.0 * float(np.mean(radii)) if radii else 1.0)

    # 5. filter small cycles
    # a cycle whose perimeter < threshold_factor * local_diameter is
    # considered a spurious junction artifact and will be collapsed.
    small_mask = np.array(perimeters) < threshold_factor * np.array(diameters)
    small_cycles = [cycles[i] for i in range(len(cycles)) if small_mask[i]]
    if not small_cycles:
        return unchanged

    # 6. group overlapping cycles (share nodes)
    # adjacent cycles (e.g. two triangles sharing a pixel) must be
    # collapsed together otherwise we leave orphan edges.
    # uses union-find to merge cycles that share at least one graph node
    small_sets = [set(c) for c in small_cycles]
    n_small = len(small_sets)
    parent = list(range(n_small))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        px, py = _find(x), _find(y)
        if px != py:
            parent[py] = px

    # pairwise check: union if cycles intersect
    for i in range(n_small):
        for j in range(i + 1, n_small):
            if small_sets[i] & small_sets[j]:
                _union(i, j)

    # collect all node IDs belonging to each "super-cycle"
    group_map: dict[int, set[int]] = {}
    for i, cs in enumerate(small_sets):
        group_map.setdefault(_find(i), set()).update(cs)

    super_cycles = list(group_map.values())

    # 7. collapse each super-cycle in the graph
    # For each super-cycle:
    #   a) identify external edges that need to be reconnected
    #   b) compute geometric centroid of all cycle node coordinates
    #   c) remove all internal edges and cycle nodes from the graph
    #   d) add a new centroid node
    #   e) reconnect external edges to the centroid

    next_node_id = max(G.nodes()) + 1
    all_removed_nodes: set[int] = set()
    # radii of voxels this step adds; every other voxel keeps its own
    added_radius: dict[tuple[int, ...], float] = {}

    for nodes in super_cycles:
        cycle_set: set[int] = set(nodes)

        # tuple content: internal_node_id, ext_node_id, edge_key (multigraph), edge_data_dict
        ext_edges: list[tuple[int, int, Any, Any]] = []
        for node in cycle_set:
            for nb in G.neighbors(node):
                if nb not in cycle_set and nb not in all_removed_nodes:
                    for key in G[node][nb]:
                        ext_edges.append((node, nb, key, G[node][nb][key]))

        # centroid = avg coordinates of all nodes clamped to image bounds
        coords_arr = np.array([node_coords[n] for n in cycle_set if n in node_coords])
        if len(coords_arr) == 0:
            continue
        centroid_coord = tuple(np.round(coords_arr.mean(axis=0)).astype(int))
        centroid_coord = tuple(
            min(max(c, 0), s - 1) for c, s in zip(centroid_coord, shape)
        )
        cycle_radii = [lookup.get(tuple(c), 0.0) for c in coords_arr]
        cycle_radii = [r for r in cycle_radii if r > 0]
        centroid_radius = float(np.mean(cycle_radii)) if cycle_radii else 0.0

        # remove edges connecting nodes within the cycle
        for u in cycle_set:
            for v in list(G.neighbors(u)):
                if v in cycle_set and G.has_edge(u, v):
                    G.remove_edge(u, v)

        # remove all cycle nodes from the graph
        for n in cycle_set:
            if G.has_node(n):
                G.remove_node(n)
        all_removed_nodes.update(cycle_set)

        # add centroid node
        cid = next_node_id
        next_node_id += 1
        G.add_node(cid)
        node_coords[cid] = centroid_coord

        # reconnect external edges to centroid
        for cycle_node, ext_node, _key, data in ext_edges:
            if not G.has_node(ext_node):
                continue

            path = data["path"]
            if len(path) < 1:
                continue

            # the path stored on the edge goes from one endpoint to the other
            # need to find out which end connects to the cycle node
            first = tuple(path[0])
            last = tuple(path[-1])
            cyc_pos = node_coords.get(cycle_node)

            if first == cyc_pos:
                # path: cycle-node -> ... -> ext-node
                # drop cycle-node pixel itself, then reverse so the
                # path goes ext-node -> ... -> adjacent-to-cycle.
                truncated = path[1:][::-1] if len(path) > 1 else path[:0]
            elif last == cyc_pos:
                # path: ext-node -> ... -> cycle-node
                # drop cycle-node pixel, keep direction.
                truncated = path[:-1] if len(path) > 1 else path[:0]
            else:
                # cycle node is somewhere in the middle of the path, search its position
                idx = None
                for i in range(len(path)):
                    if tuple(path[i]) == cyc_pos:
                        idx = i
                        break
                if idx is None:
                    continue
                else:
                    # cycle node splits the path into two arms -> keep both arms
                    # reversed on the ext side so they meet at the truncated end
                    front = path[:idx]  # ext-side front
                    back = path[idx + 1 :]  # other side
                    truncated = np.vstack([front[::-1], back])

            # determine pixel coordinate where the reconnection should start
            # (pixel closest to the old cycle node)
            if len(truncated) >= 1:
                start_coord = tuple(truncated[-1])
            elif len(path) > 1:
                start_coord = tuple(path[1])
                truncated = path[1:1]  # empty but with correct ndim
            else:
                truncated = np.empty((0, path.shape[1]), dtype=path.dtype)
                start_coord = centroid_coord  # no connection needed

            # already adjacent?
            if all(abs(a - b) <= 1 for a, b in zip(start_coord, centroid_coord)):
                full_path = truncated
            else:
                # draw line from path end to the centroid, skipping the first point of
                # the line to avoid duplicating the start_coord pixel
                line = _bresenham_line(start_coord, centroid_coord)
                line_arr = np.array(line, dtype=np.intp)
                # radii along the new line: linear from the arm's end to the centroid
                start_radius = lookup.get(start_coord, centroid_radius)
                for k, pt in enumerate(line[1:], start=1):
                    t = k / (len(line) - 1)
                    added_radius.setdefault(
                        pt, start_radius + t * (centroid_radius - start_radius)
                    )
                full_path = (
                    np.vstack([truncated, line_arr[1:]])
                    if len(line_arr) > 1
                    else truncated
                )

            if len(full_path) == 0:
                continue

            G.add_edge(ext_node, cid, path=full_path)

    # 8. the voxels of the remaining paths, with measured radii where known
    if G.number_of_edges() == 0:
        empty = np.empty((0, len(shape)), dtype=np.intp)
        return empty, np.empty(0)

    pixels = np.unique(
        np.concatenate([path for _, _, path in G.edges.data("path")]), axis=0
    )
    radii = np.array(
        [lookup.get(tuple(p), added_radius.get(tuple(p), 0.0)) for p in pixels.tolist()]
    )
    return pixels, radii


class _RadiusLookup:
    """Radius of a voxel by its coordinates, for voxels sorted by position."""

    def __init__(self, coords: np.ndarray, radius: np.ndarray, shape: tuple[int, ...]):
        self.linear = np.ravel_multi_index(np.asarray(coords, dtype=np.intp).T, shape)
        self.radius = radius
        self.shape = shape

    def get(self, coord: tuple[int, ...], default: float) -> float:
        """Radius at *coord*, or *default* if it is not one of the voxels."""
        key = np.ravel_multi_index(coord, self.shape)
        i = int(np.searchsorted(self.linear, key))
        if i < len(self.linear) and self.linear[i] == key:
            return float(self.radius[i])
        return default
