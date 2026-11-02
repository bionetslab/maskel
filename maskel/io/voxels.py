"""The voxel list: a skeleton as coordinates, radii and labels, stored in HDF5."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import h5py
import numpy as np


@dataclass
class LabelVoxels:
    """Skeleton voxels of one label, sorted by position, with its mask statistics."""

    label: int
    coords: np.ndarray
    radius: np.ndarray
    mask_voxels: int
    crop_voxels: int


@dataclass
class VoxelList:
    """Skeleton voxels of a whole image, sorted by label and then by position."""

    shape: tuple[int, ...]
    spacing: tuple[float, ...] | None
    coords: np.ndarray  # (N, ndim) global voxel indices
    radius: np.ndarray  # (N,) distance to background, in spacing units
    label: np.ndarray  # (N,)
    labels: np.ndarray  # (L,) every label of the mask, including skeleton-less ones
    mask_voxels: np.ndarray  # (L,) preprocessed foreground voxels per label
    crop_voxels: np.ndarray  # (L,) voxels of each label's padded bounding box
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        linear = np.ravel_multi_index(self.coords.T, self.shape)
        order = np.lexsort((linear, self.label))
        self.coords = self.coords[order]
        self.radius = self.radius[order]
        self.label = self.label[order]

    def __iter__(self) -> Iterator[LabelVoxels]:
        bounds = np.searchsorted(self.label, self.labels, side="left")
        ends = np.searchsorted(self.label, self.labels, side="right")
        for i, (start, stop) in enumerate(zip(bounds, ends, strict=True)):
            yield LabelVoxels(
                int(self.labels[i]),
                self.coords[start:stop],
                self.radius[start:stop],
                int(self.mask_voxels[i]),
                int(self.crop_voxels[i]),
            )

    def with_labels(self, parts: list[LabelVoxels]) -> VoxelList:
        """Copy of this list whose voxels are replaced by those of *parts*."""
        ndim = len(self.shape)
        return replace(
            self,
            coords=np.concatenate(
                [np.empty((0, ndim), np.int64)] + [p.coords for p in parts]
            ),
            radius=np.concatenate([np.empty(0)] + [p.radius for p in parts]),
            label=np.concatenate(
                [np.empty(0, np.int64)]
                + [np.full(len(p.coords), p.label, np.int64) for p in parts]
            ),
        )


def write_voxels(path: Path | str, voxels: VoxelList) -> None:
    """Write *voxels* to an HDF5 file at *path*."""
    with h5py.File(path, "w") as f:
        f["coords"] = voxels.coords
        f["radius"] = voxels.radius
        f["label"] = voxels.label
        f["labels/id"] = voxels.labels
        f["labels/mask_voxels"] = voxels.mask_voxels
        f["labels/crop_voxels"] = voxels.crop_voxels
        f.attrs["shape"] = voxels.shape
        if voxels.spacing is not None:
            f.attrs["spacing"] = voxels.spacing
        for key, value in voxels.attrs.items():
            f.attrs[key] = json.dumps(value)


def read_voxels(path: Path | str) -> VoxelList:
    """Read a voxel list written by `write_voxels`."""
    with h5py.File(path, "r") as f:
        spacing = f.attrs.get("spacing")
        return VoxelList(
            shape=tuple(int(s) for s in f.attrs["shape"]),
            spacing=None if spacing is None else tuple(float(s) for s in spacing),
            coords=f["coords"][()],
            radius=f["radius"][()],
            label=f["label"][()],
            labels=f["labels/id"][()],
            mask_voxels=f["labels/mask_voxels"][()],
            crop_voxels=f["labels/crop_voxels"][()],
            attrs={
                k: json.loads(v)
                for k, v in f.attrs.items()
                if k not in ("shape", "spacing")
            },
        )
