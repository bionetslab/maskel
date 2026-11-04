"""Splitting a volume into tile cores, each read with a halo around it."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np

DEFAULT_HALO = 100


def per_axis(value: float | tuple, ndim: int) -> tuple:
    """Expand a scalar to one value per axis, or check a tuple has *ndim* values."""
    if np.isscalar(value):
        return (value,) * ndim
    value = tuple(value)
    if len(value) != ndim:
        raise ValueError(f"expected 1 or {ndim} values, got {len(value)}")
    return value


@dataclass(frozen=True)
class Tile:
    """One tile: the core it owns and the larger block read for it."""

    core: tuple[slice, ...]
    read: tuple[slice, ...]

    @property
    def core_in_read(self) -> tuple[slice, ...]:
        """The core's position inside the read block."""
        return tuple(
            slice(c.start - r.start, c.stop - r.start)
            for c, r in zip(self.core, self.read, strict=True)
        )

    @property
    def read_size(self) -> int:
        """Number of voxels in the read block."""
        return prod(s.stop - s.start for s in self.read)


@dataclass(frozen=True)
class TileGrid:
    """Cores of *tile_shape* covering *shape*, each read with *halo* voxels per axis."""

    shape: tuple[int, ...]
    tile_shape: tuple[int, ...]
    halo: tuple[int, ...]

    @classmethod
    def build(
        cls,
        shape: tuple[int, ...],
        tile_shape: int | tuple[int, ...] | None = None,
        halo: int | tuple[int, ...] = 0,
    ) -> TileGrid:
        """Grid for *shape*; no *tile_shape* gives a single tile covering everything."""
        ndim = len(shape)
        if tile_shape is None:
            return cls(tuple(shape), tuple(shape), (0,) * ndim)
        tile_shape = per_axis(tile_shape, ndim)
        halo = per_axis(halo, ndim)
        if any(t < 1 for t in tile_shape) or any(h < 0 for h in halo):
            raise ValueError("tile_shape must be positive and halo non-negative")
        grid = cls(tuple(shape), tuple(tile_shape), tuple(halo))
        if len(grid) > 1 and grid.read_overhead > 2.0:
            warnings.warn(
                f"halo={grid.halo} with tile_shape={grid.tile_shape} reads "
                f"{grid.read_overhead:.1f}x the volume; use larger tiles.",
                stacklevel=2,
            )
        return grid

    def _spans(self, axis: int) -> list[tuple[slice, slice]]:
        """(core, read) slices of every tile along *axis*."""
        size, tile, halo = self.shape[axis], self.tile_shape[axis], self.halo[axis]
        return [
            (
                slice(start, min(start + tile, size)),
                slice(max(start - halo, 0), min(start + tile + halo, size)),
            )
            for start in range(0, size, tile)
        ]

    def __len__(self) -> int:
        return prod(len(self._spans(axis)) for axis in range(len(self.shape)))

    def __iter__(self) -> Iterator[Tile]:
        for spans in product(*(self._spans(axis) for axis in range(len(self.shape)))):
            yield Tile(tuple(c for c, _ in spans), tuple(r for _, r in spans))

    @property
    def max_read_size(self) -> int:
        """Voxels in the largest read block."""
        return prod(
            max(r.stop - r.start for _, r in self._spans(axis))
            for axis in range(len(self.shape))
        )

    @property
    def read_overhead(self) -> float:
        """Voxels read over the whole grid per voxel of volume."""
        read = prod(
            sum(r.stop - r.start for _, r in self._spans(axis))
            for axis in range(len(self.shape))
        )
        return read / prod(self.shape)
