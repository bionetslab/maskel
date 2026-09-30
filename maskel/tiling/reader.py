"""Reading blocks out of an image source without loading the whole thing."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class BlockReader:
    """A sliceable view of an image file or in-memory array.

    Files are opened lazily and memory-mapped, so `shape` costs only a header
    read and no pixel data is touched until `read` asks for a block.
    """

    def __init__(self, source: Path | str | np.ndarray) -> None:
        self.path: Path | None = None
        self._array: np.ndarray | None = None
        if isinstance(source, np.ndarray):
            self._array = source
        else:
            self.path = Path(source)

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = open_readable(self.path)
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    def read(self, slices: tuple[slice, ...]) -> np.ndarray:
        """One block as a fresh writeable array, never a view on the source."""
        return np.array(self.array[slices], order="C", copy=True)


def open_readable(path: Path) -> np.ndarray:
    """Memory-map *path*, falling back to loading it whole."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
    elif suffix in (".tif", ".tiff"):
        import tifffile

        try:
            # Only works for contiguous, uncompressed, native-endian TIFFs.
            array = tifffile.memmap(path, mode="r")
        except (ValueError, MemoryError, OSError):
            from maskel._io import load_image

            array = load_image(path)
    else:
        from maskel._io import load_image

        array = load_image(path)

    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D image, got shape={array.shape}")
    return array
