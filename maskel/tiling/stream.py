"""Source file -> thinned skeleton, one tile at a time.

The output ``.npy`` is created up front as a memory map. Each tile is read
straight out of the source, preprocessed, thinned, and written straight into
its core of the output, so neither input nor output is ever held in RAM.
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
    read_manifest_preprocess,
    write_manifest,
)
from maskel.tiling.preprocess import BlockPreprocess
from maskel.tiling.progress import ProgressReporter
from maskel.tiling.reader import BlockReader


def thin_block(
    reader: BlockReader,
    out: np.ndarray,
    spec: TileSpec,
    preprocess: BlockPreprocess | None = None,
) -> None:
    """Read one block, preprocess and thin it, write its core into *out*."""
    # The block is ours, so binarising in place is free. Both thinners
    # require a strict 0/1 input.
    block = to_binary(reader.read(spec.read), inplace=True)

    if not block.any():
        # Already zero in the output, and closing and filling cannot create
        # foreground from nothing. The thinners reject empty input anyway.
        return

    if preprocess:
        block = preprocess.apply(block)

    if block.all():
        # No background in core or halo, so the result depends on voxels we
        # did not read.
        raise ValueError(
            f"Tile {spec.index} is entirely foreground over its whole read "
            f"extent {spec.read_shape}. Increase the halo or the tile size."
        )

    out[spec.core] = lee94_thin(block)[spec.core_in_read]
    out.flush()


def _resumable_tiles(
    grid: TileGrid,
    work_dir: Path,
    out_path: Path,
    resume: bool,
    preprocess: BlockPreprocess | None,
) -> set[tuple[int, ...]]:
    """Finished tiles from a previous run, if it used this grid and output."""
    if not resume or not out_path.is_file():
        return set()
    if not (Path(work_dir) / MANIFEST_NAME).is_file():
        return set()
    wanted = preprocess.to_dict() if preprocess else None
    try:
        if read_manifest(work_dir) != grid:
            return set()
        if read_manifest_preprocess(work_dir) != wanted:
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
    preprocess: BlockPreprocess | None = None,
    progress: str | bool = "off",
    out_path: Path | None = None,
    resume: bool = True,
    dataset: str | None = None,
) -> Path:
    """Thin *source* tile by tile and return the path of the skeleton ``.npy``.

    *source* is an HDF5 file (streamed; *dataset* names the dataset inside
    it, see `BlockReader`) or an array. *halo* is what the
    thinning needs; *preprocess* adds its own margin on top. *work_dir*
    receives ``manifest.json`` and ``done.jsonl``. With *resume*, tiles already
    in the done log are skipped.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path) if out_path is not None else work_dir / SKELETON_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with BlockReader(source, dataset) as reader:
        return _skeletonize(
            reader, work_dir, out_path, tile_shape, halo, preprocess, progress, resume
        )


def _skeletonize(
    reader: BlockReader,
    work_dir: Path,
    out_path: Path,
    tile_shape: int | tuple[int, ...],
    halo: int,
    preprocess: BlockPreprocess | None,
    progress: str | bool,
    resume: bool,
) -> Path:
    margin = preprocess.margin if preprocess else 0
    grid = TileGrid(shape=reader.shape, tile_shape=tile_shape, halo=halo + margin)

    log = DoneLog(work_dir)
    already_done = _resumable_tiles(grid, work_dir, out_path, resume, preprocess)
    write_manifest(grid, work_dir, preprocess)

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
            thin_block(reader, out, spec, preprocess)
            log.record(spec)
            reporter.advance(spec)
    finally:
        # Windows keeps the file locked while a writable mapping is open.
        del out

    reporter.done()
    return out_path
