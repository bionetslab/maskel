"""Tile geometry: how the volume is split and how much each tile reads."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from math import prod

import numpy as np

DEFAULT_HALO = 100


@dataclass(frozen=True)
class TileSpec:
    """One tile: where its core is, what to read for it, and where the core
    sits inside that read block.

    All three are tuples of slices, one per axis.
    """

    index: tuple[int, ...]
    core: tuple[slice, ...]
    read: tuple[slice, ...]
    core_in_read: tuple[slice, ...]

    @property
    def name(self) -> str:
        """Filename-safe identifier, e.g. ``"0_2_1"``."""
        return "_".join(str(i) for i in self.index)

    @property
    def core_shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.core)

    @property
    def read_shape(self) -> tuple[int, ...]:
        return tuple(s.stop - s.start for s in self.read)

    def to_dict(self) -> dict[str, object]:
        """Manifest record. Bounds are half-open, in array-axis order."""
        return {
            "index": list(self.index),
            "core": _bounds(self.core),
            "read": _bounds(self.read),
            "core_in_read": _bounds(self.core_in_read),
            "core_shape": list(self.core_shape),
            "read_shape": list(self.read_shape),
        }


def _bounds(slices: tuple[slice, ...]) -> dict[str, list[int]]:
    return {"start": [s.start for s in slices], "stop": [s.stop for s in slices]}


@dataclass(frozen=True)
class TileGrid:
    """Splits *shape* into cores of *tile_shape*, each read with *halo* extra
    voxels on every side.

    A scalar *tile_shape* applies to all axes. Edge tiles are short, and the
    halo is clipped at the volume bounds.
    """

    shape: tuple[int, ...]
    tile_shape: tuple[int, ...]
    halo: int = DEFAULT_HALO

    def __post_init__(self) -> None:
        shape = tuple(int(s) for s in self.shape)
        if not shape:
            raise ValueError("shape must have at least one axis")
        if any(s < 1 for s in shape):
            raise ValueError(f"shape must be positive along every axis, got {shape}")

        raw = self.tile_shape
        tile_shape = (
            (int(raw),) * len(shape)
            if isinstance(raw, (int, np.integer))
            else tuple(int(t) for t in raw)
        )
        if len(tile_shape) != len(shape):
            raise ValueError(
                f"tile_shape has {len(tile_shape)} axes but shape has {len(shape)}"
            )
        if any(t < 1 for t in tile_shape):
            raise ValueError(f"tile_shape must be positive, got {tile_shape}")
        if int(self.halo) < 0:
            raise ValueError(f"halo must be non-negative, got {self.halo}")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "tile_shape", tile_shape)
        object.__setattr__(self, "halo", int(self.halo))

        if self.read_overhead > 2.0:
            warnings.warn(
                f"halo={self.halo} with tile_shape={tile_shape} reads "
                f"{self.read_overhead:.1f}x the volume. Use larger tiles "
                f"(or a smaller halo) to cut the redundant thinning work.",
                stacklevel=3,
            )

    @property
    def counts(self) -> tuple[int, ...]:
        """Number of tiles along each axis."""
        return tuple(-(-s // t) for s, t in zip(self.shape, self.tile_shape))

    def _axis_spans(self, axis: int) -> list[tuple[int, int, int, int]]:
        """``(core_start, core_stop, read_start, read_stop)`` per tile on *axis*."""
        tile = self.tile_shape[axis]
        extent = self.shape[axis]
        spans = []
        for i in range(self.counts[axis]):
            c0 = i * tile
            c1 = min(c0 + tile, extent)
            spans.append((c0, c1, max(0, c0 - self.halo), min(extent, c1 + self.halo)))
        return spans

    @property
    def read_total(self) -> int:
        """Voxels read over the whole grid. Separable per axis, so no need to
        walk every tile."""
        return prod(
            sum(r1 - r0 for _, _, r0, r1 in self._axis_spans(axis))
            for axis in range(len(self.shape))
        )

    @property
    def read_overhead(self) -> float:
        """`read_total` per voxel of volume. Always >= 1."""
        return self.read_total / prod(self.shape)

    def __len__(self) -> int:
        return prod(self.counts)

    def __iter__(self) -> Iterator[TileSpec]:
        axis_spans = [self._axis_spans(axis) for axis in range(len(self.shape))]
        for index in product(*(range(n) for n in self.counts)):
            core, read, core_in_read = [], [], []
            for axis, i in enumerate(index):
                c0, c1, r0, r1 = axis_spans[axis][i]
                core.append(slice(c0, c1))
                read.append(slice(r0, r1))
                core_in_read.append(slice(c0 - r0, c1 - r0))
            yield TileSpec(index, tuple(core), tuple(read), tuple(core_in_read))

    def to_dict(self) -> dict[str, object]:
        """Grid parameters plus every tile's coordinates. The scalars are the
        source of truth; the tile list is for reading and external tools."""
        return {
            "shape": list(self.shape),
            "tile_shape": list(self.tile_shape),
            "halo": self.halo,
            "num_tiles": len(self),
            "read_overhead": round(self.read_overhead, 4),
            "tiles": [spec.to_dict() for spec in self],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TileGrid:
        grid = cls(
            shape=tuple(data["shape"]),  # type: ignore[arg-type]
            tile_shape=tuple(data["tile_shape"]),  # type: ignore[arg-type]
            halo=int(data["halo"]),  # type: ignore[arg-type]
        )
        # Catches a hand-edited or truncated manifest.
        tiles = data.get("tiles")
        if tiles is not None and len(tiles) != len(grid):  # type: ignore[arg-type]
            raise ValueError(
                f"Manifest lists {len(tiles)} tiles but its parameters "  # type: ignore[arg-type]
                f"describe {len(grid)}."
            )
        return grid
