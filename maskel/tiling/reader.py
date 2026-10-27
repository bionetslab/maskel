"""Reading blocks out of an HDF5 dataset without loading the whole thing."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np

HDF5_EXTENSIONS = frozenset({".h5", ".hdf5"})


class BlockReader:
    """A sliceable view of an HDF5 dataset or an in-memory array.

    Files are opened lazily, and HDF5 reads only the chunks a block overlaps,
    so `shape` costs only a header read and no pixel data is touched until
    `read` asks for a block. Compressed and chunked datasets stream as well as
    contiguous ones.

    *dataset* is the path of the dataset inside the file. It may be omitted
    when the file holds exactly one 2D or 3D dataset.

    Use as a context manager, or call `close`, to release the file.
    """

    def __init__(
        self, source: Path | str | np.ndarray, dataset: str | None = None
    ) -> None:
        self.path: Path | None = None
        self.dataset_name = dataset
        self._file = None
        self._data = None
        if isinstance(source, np.ndarray):
            if dataset is not None:
                raise ValueError("dataset only applies to an HDF5 file")
            self._data = source
        else:
            self.path = Path(source)
            if self.path.suffix.lower() not in HDF5_EXTENSIONS:
                raise ValueError(
                    f"Tiled input must be an HDF5 file "
                    f"({', '.join(sorted(HDF5_EXTENSIONS))}), got "
                    f"'{self.path.name}'. Other formats cannot be read block "
                    f"by block; convert the image to HDF5 first."
                )

    @property
    def data(self):
        """The underlying array or `h5py.Dataset`. Slicing it reads from disk."""
        if self._data is None:
            self._file, self._data = open_hdf5(self.path, self.dataset_name)
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    def read(self, slices: tuple[slice, ...]) -> np.ndarray:
        """One block as a fresh writeable array, never a view on the source."""
        return np.array(self.data[slices], order="C", copy=True)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._data = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def open_hdf5(path: Path, dataset: str | None = None):
    """Open *path* read-only and return ``(file, dataset)``.

    Raises instead of falling back to loading anything whole: a dataset that
    cannot be found, is not 2D/3D, or uses a compression filter that is not
    available fails here, before any tile is processed.
    """
    import h5py

    try:
        # Registers Blosc, LZ4, Zstd, ... with HDF5 when installed.
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass

    path = Path(path)
    f = h5py.File(path, "r")
    try:
        if dataset is None:
            candidates = _image_datasets(f)
            if len(candidates) != 1:
                found = ", ".join(candidates) or "none"
                raise ValueError(
                    f"'{path.name}' must hold exactly one 2D or 3D dataset when "
                    f"no dataset is given; found: {found}. Pick one with "
                    f"--dataset."
                )
            (dataset,) = candidates
        if dataset not in f or not isinstance(f[dataset], h5py.Dataset):
            raise ValueError(f"'{path.name}' has no dataset '{dataset}'.")
        data = f[dataset]

        if data.ndim not in (2, 3):
            raise ValueError(
                f"Expected a 2D or 3D dataset, got '{dataset}' with shape={data.shape}"
            )
        if not (np.issubdtype(data.dtype, np.number) or data.dtype == np.bool_):
            raise ValueError(f"Dataset '{dataset}' has non-numeric dtype {data.dtype}.")
        if data.size == 0:
            raise ValueError(f"Dataset '{dataset}' is empty.")

        # A missing compression filter only shows up on the first read, so
        # touch one voxel now rather than failing hours into the run.
        try:
            data[(slice(0, 1),) * data.ndim]
        except OSError as exc:
            raise ValueError(
                f"Cannot decode dataset '{dataset}' in '{path.name}': {exc}. "
                f"Its compression filter is probably not installed; try "
                f"`pip install hdf5plugin`."
            ) from exc
    except BaseException:
        f.close()
        raise
    return f, data


def _image_datasets(f) -> list[str]:
    """Paths of every 2D or 3D dataset in the file."""
    import h5py

    found: list[str] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
            found.append(name)

    f.visititems(visit)
    return found
