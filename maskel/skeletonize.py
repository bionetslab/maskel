"""Stage 1: mask -> voxel list, tile by tile and label by label."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import warnings
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
from scipy import ndimage as ndi

from maskel.config import PreprocessingConfig
from maskel.io.images import open_hdf5
from maskel.io.voxels import VoxelList
from maskel.preprocessing import crop_pad, preprocess_binary, preprocessing_margin
from maskel.progress import ProgressReporter
from maskel.radii import skeleton_radii
from maskel.thin import lee94_thin
from maskel.tiling import DEFAULT_HALO, Tile, TileGrid, per_axis

# Halo the thinning needs per voxel of radius. Lee94 has no strict bound; measured
# needs were 0.5-2.1x the largest radius (tubes, blobs, subcrop2), so 4x leaves margin.
HALO_PER_RADIUS = 4

# Peak memory of one tile beyond its input block, measured at 7-11 bytes per read
# voxel (4-50 % foreground), plus what a fresh worker process starts with.
TILE_BYTES_PER_VOXEL = 12
PROCESS_BYTES = 300 * 2**20


@dataclass
class _LabelStats:
    """Preprocessed foreground count and raw bounding box of one label, over all tiles."""

    mask_voxels: int = 0
    start: np.ndarray | None = None
    stop: np.ndarray | None = None

    def add(self, mask_voxels: int, start: np.ndarray, stop: np.ndarray) -> None:
        """Count one tile's share of the label."""
        self.mask_voxels += mask_voxels
        self.start = start if self.start is None else np.minimum(self.start, start)
        self.stop = stop if self.stop is None else np.maximum(self.stop, stop)


def skeletonize(
    image,
    preprocessing: PreprocessingConfig,
    *,
    spacing: tuple[float, ...] | None = None,
    tile_shape: int | tuple[int, ...] | None = None,
    halo: int | tuple[int, ...] = DEFAULT_HALO,
    progress: str | bool = "off",
    jobs: int = 1,
    source: tuple[Path, str | None] | None = None,
) -> VoxelList:
    """Preprocess, thin and measure every label of *image*, tiles in *jobs* processes reading *source* (HDF5 path, dataset)."""
    ndim = image.ndim
    halo = per_axis(halo, ndim)
    margin = preprocessing_margin(preprocessing, spacing, ndim)
    grid = TileGrid.build(
        image.shape, tile_shape, tuple(h + m for h, m in zip(halo, margin))
    )
    if (
        len(grid) > 1
        and preprocessing.fill_holes
        and preprocessing.max_hole_diameter <= 0
    ):
        raise ValueError(
            "fill_holes without max_hole_diameter only works on a single tile."
        )
    jobs = _usable_jobs(jobs, len(grid))
    if jobs > 1 and source is None:
        raise ValueError("Parallel tiles need an HDF5 input to read blocks from.")
    _check_memory(grid, image.dtype, jobs)

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    stats: dict[int, _LabelStats] = {}
    reporter = ProgressReporter("Skeletonizing", grid, progress)
    args = (preprocessing, spacing, len(grid) > 1)
    warned = np.zeros(ndim, dtype=int)  # largest halo a warning already asked for
    for tile, (tile_parts, tile_stats) in _tile_results(
        image, grid, args, jobs, source
    ):
        parts.extend(tile_parts)
        for label, counts in tile_stats.items():
            stats.setdefault(label, _LabelStats()).add(*counts)
        if len(grid) > 1:
            warned = _warn_small_halo(tile, tile_parts, halo, spacing, warned)
        reporter.advance(tile)
    reporter.done()

    return _assemble(image.shape, spacing, parts, stats, crop_pad(preprocessing))


def _usable_jobs(jobs: int, num_tiles: int) -> int:
    """*jobs*, capped with a warning at the number of tiles and CPU cores."""
    cores = os.cpu_count() or 1
    if jobs > min(num_tiles, cores):
        warnings.warn(
            f"--jobs {jobs} capped to {min(num_tiles, cores)}: there are "
            f"{num_tiles} tile(s) and {cores} CPU core(s).",
            stacklevel=3,
        )
    return max(1, min(jobs, num_tiles, cores))


def _check_memory(grid: TileGrid, dtype: np.dtype, jobs: int) -> None:
    """Fail before the run when *jobs* tiles at once would not fit in free memory."""
    input_bytes = dtype.itemsize + (0 if np.issubdtype(dtype, np.integer) else 8)
    per_tile = grid.max_read_size * (input_bytes + TILE_BYTES_PER_VOXEL)
    needed = jobs * per_tile + (PROCESS_BYTES * jobs if jobs > 1 else 0)
    available = psutil.virtual_memory().available
    if needed > available:
        raise MemoryError(
            f"{jobs} tile(s) of up to {grid.max_read_size:,} voxels need about "
            f"{needed / 2**30:.1f} GiB, but only {available / 2**30:.1f} GiB is "
            "free; use smaller tiles or fewer jobs."
        )


def _tile_results(
    image,
    grid: TileGrid,
    args: tuple,
    jobs: int,
    source: tuple[Path, str | None] | None,
) -> Iterator[tuple[Tile, tuple[list, dict]]]:
    """Each tile with its `_skeletonize_tile` result, in order of completion."""
    if jobs <= 1:
        for tile in grid:
            yield tile, _skeletonize_tile(np.asarray(image[tile.read]), tile, *args)
        return

    threads = max(1, (os.cpu_count() or 1) // jobs)
    pool = ProcessPoolExecutor(
        jobs,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(*source, threads),
    )
    futures = {
        pool.submit(_skeletonize_tile_in_worker, tile, *args): tile for tile in grid
    }
    try:
        for future in as_completed(futures):
            yield futures[future], future.result()
    except BaseException:
        # A failed tile or Ctrl-C: stop the other workers instead of waiting.
        workers = mp.active_children()
        pool.shutdown(wait=False, cancel_futures=True)
        for worker in workers:
            worker.kill()
        raise
    pool.shutdown()


_worker_file = _worker_image = None


def _init_worker(path: Path, dataset: str | None, threads: int) -> None:
    """Open the input once per worker and share the cores between workers."""
    global _worker_file, _worker_image
    import numba

    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    # Kept open for the worker's lifetime; it exits when the pool shuts down.
    _worker_file, _worker_image = open_hdf5(path, dataset)


def _skeletonize_tile_in_worker(tile: Tile, *args) -> tuple[list, dict]:
    """`_skeletonize_tile` on a block read from the worker's input."""
    return _skeletonize_tile(np.asarray(_worker_image[tile.read]), tile, *args)


def _skeletonize_tile(
    block: np.ndarray,
    tile: Tile,
    preprocessing: PreprocessingConfig,
    spacing: tuple[float, ...] | None,
    tiled: bool,
) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray]], dict[int, tuple]]:
    """Skeleton voxels (coords, radius, label) and per-label statistics of *tile*'s core."""
    if not np.issubdtype(block.dtype, np.integer):
        block = block.astype(np.int64)
    core = tile.core_in_read
    read_start = np.array([s.start for s in tile.read])
    core_lo = np.array([s.start for s in core])
    core_hi = np.array([s.stop for s in core])

    in_core = ndi.find_objects(block[core])
    boxes = ndi.find_objects(block)
    pad = crop_pad(preprocessing)
    parts = []
    stats = {}  # label -> (mask voxels, bbox start, bbox stop) in the core
    for index, core_box in enumerate(in_core):
        if core_box is None:
            continue
        label = index + 1
        crop = tuple(
            slice(max(s.start - pad, 0), min(s.stop + pad, n))
            for s, n in zip(boxes[index], block.shape, strict=True)
        )
        crop_start = np.array([s.start for s in crop])
        binary = preprocess_binary(
            (block[crop] == label).astype(np.uint8),
            opening_radius=preprocessing.opening_radius,
            closing_iterations=preprocessing.closing_iterations,
            fill_holes=preprocessing.fill_holes,
            max_hole_diameter=preprocessing.max_hole_diameter,
            spacing=spacing,
        )

        # Mask statistics count only the core, so each voxel counts once.
        core_in_crop = tuple(
            slice(max(lo - c, 0), max(hi - c, 0))
            for lo, hi, c in zip(core_lo, core_hi, crop_start, strict=True)
        )
        stats[label] = (
            int(np.count_nonzero(binary[core_in_crop])),
            read_start + core_lo + [s.start for s in core_box],
            read_start + core_lo + [s.stop for s in core_box],
        )

        if not binary.any():
            continue
        if tiled and binary.all():
            # No background anywhere, so the result depends on unread voxels.
            raise ValueError(
                f"Label {label} fills its whole block {binary.shape}; "
                "increase the halo or the tile size."
            )
        coords = np.argwhere(lee94_thin(binary))
        radius = skeleton_radii(binary, coords, spacing)
        coords += crop_start
        keep = np.all((coords >= core_lo) & (coords < core_hi), axis=1)
        parts.append(
            (
                coords[keep] + read_start,
                radius[keep],
                np.full(int(keep.sum()), label, np.int64),
            )
        )
    return parts, stats


def _assemble(
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None,
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    stats: dict[int, _LabelStats],
    pad: int,
) -> VoxelList:
    """Combine the per-tile parts into one voxel list; a single label is renamed 1."""
    ndim = len(shape)
    labels = np.array(sorted(stats), dtype=np.int64)
    coords = np.concatenate([np.empty((0, ndim), np.int64)] + [p[0] for p in parts])
    radius = np.concatenate([np.empty(0)] + [p[1] for p in parts])
    label = np.concatenate([np.empty(0, np.int64)] + [p[2] for p in parts])
    crop_voxels = [
        math.prod(
            min(stop + pad, n) - max(start - pad, 0)
            for start, stop, n in zip(stats[i].start, stats[i].stop, shape, strict=True)
        )
        for i in labels
    ]
    if len(labels) == 1:
        label[:] = 1
        labels = np.array([1], dtype=np.int64)
    return VoxelList(
        shape=tuple(shape),
        spacing=spacing,
        coords=coords.astype(np.int64),
        radius=radius,
        label=label,
        labels=labels,
        mask_voxels=np.array([stats[i].mask_voxels for i in sorted(stats)], np.int64),
        crop_voxels=np.array(crop_voxels, dtype=np.int64),
    )


def _warn_small_halo(
    tile: Tile,
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    halo: tuple[int, ...],
    spacing: tuple[float, ...] | None,
    warned: np.ndarray,
) -> np.ndarray:
    """Warn when *tile*'s largest radius needs more halo than given; return the largest need warned about."""
    radius = max((p[1].max() for p in parts if len(p[1])), default=0.0)
    step = np.ones(len(halo)) if spacing is None else np.asarray(spacing)
    needed = np.ceil(HALO_PER_RADIUS * radius / step).astype(int)
    if ((needed > np.array(halo)) & (needed > warned)).any():
        start = tuple(s.start for s in tile.core)
        warnings.warn(
            f"Tile at {start}: largest radius {radius:.3g} needs a halo of about "
            f"{tuple(needed.tolist())} voxels, got {halo}; tile seams may differ "
            "from an untiled run.",
            stacklevel=3,
        )
        return np.maximum(warned, needed)
    return warned
