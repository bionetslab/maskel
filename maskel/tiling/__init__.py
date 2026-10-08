"""Tiled (out-of-core) skeletonization.

Thinning is a local operation, so a tile can be thinned on its own as long as
it is read with a halo wider than the algorithm's reach. Only the halo-free
core of each thinned tile is kept, and the cores tile the volume exactly.

Closing and hole filling are local too, given enough halo, so they run on each
block just before its thinning - no extra pass, no intermediate volume::

    grid halo = thinning halo + 2 * closing_iterations + max_hole_size

Hole filling needs `max_hole_size`, because an unbounded hole can span the
whole volume. It can never over-fill: a block-enclosed background region is
provably enclosed globally too.

Halo for the thinning: roughly 8x the largest vessel radius in 3D. Too small
fails silently, as small differences near tile seams.

Memory: one tile at a time, about 5 bytes per voxel of the read extent
(``tile + 2 * halo`` per axis) plus 7 bytes per foreground voxel.

`TileGrid.read_overhead` gives the redundant-work multiplier - the halo is
thinned too, then discarded.

Nothing after the thinning runs here; `analyze_tiled` rejects those steps.
"""

from maskel.tiling.analyze import analyze_tiled, preprocess_from_config
from maskel.tiling.grid import DEFAULT_HALO, TileGrid, TileSpec
from maskel.tiling.manifest import (
    DONE_LOG_NAME,
    MANIFEST_NAME,
    SKELETON_NAME,
    DoneLog,
    read_manifest,
    write_manifest,
)
from maskel.tiling.preprocess import BlockPreprocess
from maskel.tiling.progress import PROGRESS_ENV_VAR, PROGRESS_STYLES
from maskel.tiling.reader import BlockReader
from maskel.tiling.stream import skeletonize_streaming

__all__ = [
    "DEFAULT_HALO",
    "DONE_LOG_NAME",
    "MANIFEST_NAME",
    "PROGRESS_ENV_VAR",
    "PROGRESS_STYLES",
    "SKELETON_NAME",
    "BlockPreprocess",
    "BlockReader",
    "DoneLog",
    "TileGrid",
    "TileSpec",
    "analyze_tiled",
    "preprocess_from_config",
    "read_manifest",
    "skeletonize_streaming",
    "write_manifest",
]
