"""Source file -> thinned skeleton, one tile at a time.

The output ``.npy`` is created up front as a memory map. Each tile is read
straight out of the source, thinned, and written straight into its core of
the output, so neither input nor output is ever held in RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from maskel._utils import to_binary
from maskel.thin import lee94_thin
from maskel.tiling.grid import DEFAULT_HALO, TileGrid, TileSpec
from maskel.tiling.manifest import (
    MANIFEST_NAME,
    SKELETON_NAME,
    DoneLog,
    read_manifest,
    write_manifest,
)
from maskel.tiling.progress import ProgressReporter
from maskel.tiling.reader import BlockReader


def thin_block(reader: BlockReader, out: np.ndarray, spec: TileSpec) -> None:
    """Read one block, thin it, write its core into *out*."""
    # The block is ours, so binarising in place is free. Both thinners
    # require a strict 0/1 input.
    block = to_binary(reader.read(spec.read), inplace=True)

    foreground = int(np.count_nonzero(block))
    if foreground == 0:
        # Already zero in the output, and the thinners reject empty input.
        return
    if foreground == block.size:
        # No background in core or halo, so the result depends on voxels we
        # did not read.
        raise ValueError(
            f"Tile {spec.index} is entirely foreground over its whole read "
            f"extent {spec.read_shape}. Increase the halo or the tile size."
        )

    out[spec.core] = lee94_thin(block)[spec.core_in_read]
    out.flush()


def _resumable_tiles(
    grid: TileGrid, work_dir: Path, out_path: Path, resume: bool
) -> set[tuple[int, ...]]:
    """Finished tiles from a previous run, if it used this grid and output."""
    if not resume or not out_path.is_file():
        return set()
    if not (Path(work_dir) / MANIFEST_NAME).is_file():
        return set()
    try:
        if read_manifest(work_dir) != grid:
            return set()
    except (ValueError, KeyError, TypeError):
        return set()

    existing = np.load(out_path, mmap_mode="r")
    matches = tuple(existing.shape) == grid.shape and existing.dtype == np.uint8
    del existing
    if not matches:
        return set()

    return DoneLog(work_dir).read()


def skeletonize_streaming(
    source: Path | str | np.ndarray,
    work_dir: Path,
    *,
    tile_shape: int | tuple[int, ...],
    halo: int = DEFAULT_HALO,
    progress: str | bool = "off",
    out_path: Path | None = None,
    resume: bool = True,
) -> Path:
    """Thin *source* tile by tile and return the path of the skeleton ``.npy``.

    *source* is an image path (streamed) or an array. *work_dir* receives
    ``manifest.json`` and ``done.jsonl``. With *resume*, tiles already in the
    done log are skipped.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path) if out_path is not None else work_dir / SKELETON_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reader = BlockReader(source)
    grid = TileGrid(shape=reader.shape, tile_shape=tile_shape, halo=halo)

    log = DoneLog(work_dir)
    already_done = _resumable_tiles(grid, work_dir, out_path, resume)
    write_manifest(grid, work_dir)

    if already_done:
        # Reuse the previous output; recreating it would zero finished cores.
        out = np.lib.format.open_memmap(out_path, mode="r+")
    else:
        log.clear()
        out = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.uint8, shape=grid.shape
        )

    reporter = ProgressReporter("Thinning", grid, progress)
    reporter.skip(spec for spec in grid if spec.index in already_done)

    try:
        for spec in grid:
            if spec.index in already_done:
                continue
            thin_block(reader, out, spec)
            log.record(spec)
            reporter.advance(spec)
    finally:
        # Windows keeps the file locked while a writable mapping is open.
        del out

    reporter.done()
    return out_path
