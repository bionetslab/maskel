"""Tiled (out-of-core) skeletonization.

Lee94 thinning is a local operation: whether a voxel gets removed depends only
on its 3x3x3 neighbourhood, so influence propagates at most one voxel per
sub-iteration. A tile can therefore be thinned independently of its neighbours,
provided it is *read* with a halo thicker than the algorithm's domain of
dependence. Only the halo-free **core** of each thinned tile is kept.

Thinning is all this does. Preprocessing, junction cleanup, radii, the graph
and the features are whole-volume operations and stay in
`vesskel.pipeline.analyze_binary_image`; `analyze_tiled` raises rather than
silently skipping any of them that the config asks for. That narrowness is
what makes the tiled skeleton bit-identical to the monolithic one whenever
the halo is adequate.

Choosing the halo
-----------------
The number of sub-iterations scales with the thickest structure present: it
takes about one sweep per voxel of half-thickness to peel a structure down to
its centreline, and a 3D sweep is six sub-iterations (one per face). So the
dependence radius is roughly ``6 * max_radius``, and 3D is far hungrier than
2D, which sweeps only four directions.

Measured on `tests/test_tiling.py` fixtures - a structure 4 voxels thick
(radius 2) needed halo 16 in 3D to reproduce monolithic thinning exactly,
while a 6-voxel-thick 2D image converged at halo 2. As a rule of thumb use
``halo >= 8 * max_radius`` for 3D; `DEFAULT_HALO` covers radii up to about 12
voxels. When in doubt, verify rather than assume: rerun a sample of tiles at
double the halo and compare the cores. An undersized halo fails silently, as
small local differences near tile seams.

The cores partition the volume exactly (they never overlap and leave no gaps),
so writing a thinned core into the output is a plain assignment - there is no
overlap to blend or resolve, and disjoint cores are why parallel workers can
share one output map.

Memory and I/O
--------------
Peak memory is one tile, and it is set by the tile's *read* extent
(``tile + 2 * halo`` per axis), not by its core. Thinning holds roughly
**5 bytes per read voxel** - the block, `thin_3d`'s binarised copy, its padded
working array, its ``removed_epoch`` stamps and the returned copy - plus about
7 bytes per *foreground* voxel for the candidate list. Budget against that,
and multiply by ``jobs``.

The halo also costs redundant work: each axis reads up to ``tile + 2 * halo``
voxels per tile, and the cost compounds across axes. `TileGrid.read_overhead`
reports the exact multiplier - and note it multiplies the *thinning*, not just
the bytes read, since the halo is thinned too and then discarded. Keep tile
edges well above the halo to hold it down. An axis shorter than the tile
contributes no overhead at all, since it is never split.

Total disk traffic is ``read_overhead`` x the volume in reads, plus one dense
write of the output. Nothing is staged in between.
"""

from vesskel.tiling.analyze import analyze_tiled
from vesskel.tiling.grid import DEFAULT_HALO, TileGrid, TileSpec
from vesskel.tiling.manifest import (
    DONE_LOG_NAME,
    MANIFEST_NAME,
    SKELETON_NAME,
    DoneLog,
    read_manifest,
    write_manifest,
)
from vesskel.tiling.progress import (
    PROGRESS_ENV_VAR,
    PROGRESS_STYLES,
    ProgressReporter,
)
from vesskel.tiling.reader import BlockReader
from vesskel.tiling.stream import skeletonize_streaming, thin_block

__all__ = [
    "DEFAULT_HALO",
    "DONE_LOG_NAME",
    "MANIFEST_NAME",
    "PROGRESS_ENV_VAR",
    "PROGRESS_STYLES",
    "SKELETON_NAME",
    "BlockReader",
    "DoneLog",
    "ProgressReporter",
    "TileGrid",
    "TileSpec",
    "analyze_tiled",
    "read_manifest",
    "skeletonize_streaming",
    "thin_block",
    "write_manifest",
]
