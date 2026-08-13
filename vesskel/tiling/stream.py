"""The streaming driver: source file -> thinned skeleton, one tile at a time.

Nothing is staged. The output ``.npy`` is created up front as a memory map,
and each tile is read straight out of the source, thinned, and written
straight into its core of the output. Neither the input nor the output is
ever held in RAM - peak memory is one tile.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from vesskel._utils import to_binary
from vesskel.thin import lee94_thin
from vesskel.tiling.grid import DEFAULT_HALO, TileGrid, TileSpec
from vesskel.tiling.manifest import (
    MANIFEST_NAME,
    SKELETON_NAME,
    DoneLog,
    read_manifest,
    write_manifest,
)
from vesskel.tiling.progress import ProgressReporter
from vesskel.tiling.reader import BlockReader


def thin_block(reader: BlockReader, out_path: Path, spec: TileSpec) -> None:
    """Read one block, thin it, write its core into the output map.

    Top-level (not a closure) so it survives pickling to spawned workers.
    Each call opens its own ``r+`` mapping: cores are disjoint, so concurrent
    workers never write the same byte, and mappings of one file are coherent.
    """
    # `read` hands back a block we own, so binarising in place costs nothing.
    # `thin_3d` would binarise internally anyway, but `thin_2d` expects a
    # strict 0/1 input, so the conversion has to happen here.
    skeleton = lee94_thin(to_binary(reader.read(spec.read), inplace=True))

    out = np.lib.format.open_memmap(out_path, mode="r+")
    try:
        out[spec.core] = skeleton[spec.core_in_read]
        out.flush()
    finally:
        # Release the mapping; on Windows a lingering writable handle keeps
        # the file locked against later readers.
        del out


def _resumable_tiles(
    grid: TileGrid, work_dir: Path, out_path: Path, resume: bool
) -> set[tuple[int, ...]]:
    """Which tiles a previous run already finished, if they can be trusted.

    Resuming is only valid when that run used the same grid and left a
    correctly shaped output behind. Otherwise the done log describes work
    that no longer corresponds to this output, and everything is redone.
    """
    if not resume or not out_path.is_file():
        return set()

    if not (Path(work_dir) / MANIFEST_NAME).is_file():
        return set()
    try:
        if read_manifest(work_dir) != grid:
            return set()
    except (ValueError, KeyError, TypeError):  # JSONDecodeError is a ValueError
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
    jobs: int = 1,
    progress: str | bool = "off",
    out_path: Path | None = None,
    resume: bool = True,
) -> Path:
    """Thin *source* tile by tile, straight from disk to disk.

    Thinning is the only thing applied. There is no preprocessing and no
    postprocessing - see `vesskel.tiling.analyze_tiled` for the config-aware
    entry point, which rejects the steps this cannot honour.

    Parameters
    ----------
    source : Path | str | ndarray
        Image path (streamed) or an in-memory array (convenient for small
        inputs and tests).
    work_dir : Path
        Receives ``manifest.json`` and ``done.jsonl``. No per-tile files.
    tile_shape : int | tuple[int, ...]
        Core edge length(s). This sets peak memory; see `vesskel.tiling`.
    halo : int
        Voxels read beyond each core. Must exceed the thinning's domain of
        dependence or the result differs near tile seams, silently.
    jobs : int
        Worker processes. 1 runs in-process; 0 uses every CPU core. Each
        worker holds its own tile, so memory scales with this.
    resume : bool
        Skip tiles already recorded in ``done.jsonl``, provided the manifest
        and output on disk match this run. On by default: these runs are long
        enough that starting over after a crash is a real cost.

    Returns
    -------
    Path
        The written skeleton ``.npy``.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path) if out_path is not None else work_dir / SKELETON_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = (os.cpu_count() or 1) if jobs == 0 else jobs
    if jobs < 1:
        raise ValueError("jobs must be 0 (auto) or a positive integer")

    reader = BlockReader(source)
    grid = TileGrid(shape=reader.shape, tile_shape=tile_shape, halo=halo)

    log = DoneLog(work_dir)
    already_done = _resumable_tiles(grid, work_dir, out_path, resume)
    write_manifest(grid, work_dir)

    if already_done:
        # Reopen the previous output in place; recreating it would zero the
        # cores those skipped tiles already hold.
        handle = np.lib.format.open_memmap(out_path, mode="r+")
    else:
        log.clear()
        handle = np.lib.format.open_memmap(
            out_path, mode="w+", dtype=np.uint8, shape=grid.shape
        )
    del handle

    pending = [spec for spec in grid if spec.index not in already_done]
    reporter = ProgressReporter("Thinning", grid, progress)
    reporter.skip(spec for spec in grid if spec.index in already_done)

    # The parent is the only writer of the done log, so records never
    # interleave and a line is only appended once the tile is really on disk.
    with log:
        if jobs == 1 or len(pending) <= 1:
            for spec in pending:
                thin_block(reader, out_path, spec)
                log.record(spec)
                reporter.advance(spec)
        else:
            _run_parallel(reader, out_path, pending, jobs, log, reporter)

    reporter.done()
    return out_path


def _run_parallel(
    reader: BlockReader,
    out_path: Path,
    pending: list[TileSpec],
    jobs: int,
    log: DoneLog,
    reporter: ProgressReporter,
) -> None:
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
        futures = {ex.submit(thin_block, reader, out_path, s): s for s in pending}
        # Wall-clock throughput already reflects all workers, so the same
        # elapsed/done extrapolation holds; only the completion order differs.
        for fut in as_completed(futures):
            fut.result()  # re-raise worker exceptions here
            spec = futures[fut]
            log.record(spec)
            reporter.advance(spec)
