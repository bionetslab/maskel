"""Block-wise access to an image source, without loading it."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class BlockReader:
    """A sliceable, lazily-opened view of an image file or in-memory array.

    Opening is deferred so the object can be pickled to a spawned worker,
    which then maps the file itself. Nothing but the path crosses the process
    boundary.
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
        """Materialise one block. Only these pages are touched on disk.

        Always a fresh, writeable, C-contiguous array the caller owns, even
        when the requested slice happens to span the whole source - callers
        binarise the block in place, and aliasing an in-memory source there
        would corrupt the caller's input.
        """
        return np.array(self.array[slices], order="C", copy=True)

    def __getstate__(self) -> dict[str, object]:
        if self.path is None:
            # An in-memory source has to travel; only small inputs use one.
            return {"path": None, "array": self._array}
        return {"path": self.path, "array": None}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.path = state["path"]
        self._array = state["array"]


def open_readable(path: Path) -> np.ndarray:
    """Map *path* without reading it, falling back to a full load."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, mmap_mode="r")
    elif suffix in (".tif", ".tiff"):
        import tifffile

        try:
            # Works for contiguous, uncompressed, native-endian TIFFs - which
            # is what a raw volume dump is. Anything else raises and we load.
            array = tifffile.memmap(path, mode="r")
        except (ValueError, MemoryError, OSError):
            from vesskel._io import load_image

            array = load_image(path)
    else:
        from vesskel._io import load_image

        array = load_image(path)

    if array.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D image, got shape={array.shape}")
    return array
