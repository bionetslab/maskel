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
from scipy import ndimage as ndi

from maskel.config import ExtractionConfig
from maskel.io.images import open_hdf5
from maskel.io.voxels import VoxelList
from maskel.preprocessing import crop_pad, preprocess_binary, preprocessing_margin
from maskel.progress import ProgressReporter
from maskel.radii import skeleton_radii
from maskel.thin import lee94_thin
from maskel.tiling import DEFAULT_HALO, Tile, TileGrid, per_axis

# Halo the thinning needs per voxel of radius (measured once: radius 2 needed 16).
HALO_PER_RADIUS = 8


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
    extraction: ExtractionConfig,
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
    margin = preprocessing_margin(extraction, spacing, ndim)
    grid = TileGrid.build(
        image.shape, tile_shape, tuple(h + m for h, m in zip(halo, margin))
    )
    if len(grid) > 1 and extraction.fill_holes and extraction.max_hole_diameter <= 0:
        raise ValueError(
            "fill_holes without max_hole_diameter only works on a single tile."
        )

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    stats: dict[int, _LabelStats] = {}
    reporter = ProgressReporter("Skeletonizing", grid, progress)
    args = (extraction, spacing, len(grid) > 1)
    for tile, (tile_parts, tile_stats) in _tile_results(
        image, grid, args, jobs, source
    ):
        parts.extend(tile_parts)
        for label, counts in tile_stats.items():
            stats.setdefault(label, _LabelStats()).add(*counts)
        reporter.advance(tile)
    reporter.done()

    voxels = _assemble(image.shape, spacing, parts, stats, crop_pad(extraction))
    if len(grid) > 1:
        _warn_small_halo(voxels.radius, halo, spacing)
    return voxels


def _tile_results(
    image,
    grid: TileGrid,
    args: tuple,
    jobs: int,
    source: tuple[Path, str | None] | None,
) -> Iterator[tuple[Tile, tuple[list, dict]]]:
    """Each tile with its `_skeletonize_tile` result, in order of completion."""
    if jobs <= 1 or len(grid) == 1:
        for tile in grid:
            yield tile, _skeletonize_tile(np.asarray(image[tile.read]), tile, *args)
        return
    if source is None:
        raise ValueError("Parallel tiles need an HDF5 input to read blocks from.")

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
    extraction: ExtractionConfig,
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
    pad = crop_pad(extraction)
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
            opening_radius=extraction.opening_radius,
            closing_iterations=extraction.closing_iterations,
            fill_holes=extraction.fill_holes,
            max_hole_diameter=extraction.max_hole_diameter,
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
    radius: np.ndarray, halo: tuple[int, ...], spacing: tuple[float, ...] | None
) -> None:
    """Warn when the largest radius needs more halo than given on some axis."""
    if radius.size == 0:
        return
    step = np.ones(len(halo)) if spacing is None else np.asarray(spacing)
    needed = np.ceil(HALO_PER_RADIUS * radius.max() / step).astype(int)
    if (needed > np.array(halo)).any():
        warnings.warn(
            f"Largest radius {radius.max():.3g} needs a halo of about "
            f"{tuple(needed.tolist())} voxels, got {halo}; tile seams may differ "
            "from an untiled run.",
            stacklevel=3,
        )
