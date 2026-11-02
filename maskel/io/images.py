"""Opening input images: HDF5 lazily, every other format loaded whole."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import itk
import numpy as np
import tifffile
from PIL import Image

HDF5_EXTENSIONS = frozenset({".h5", ".hdf5"})
IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy", ".mhd"}
)


@contextmanager
def open_image(path: Path | str, dataset: str | None = None) -> Iterator:
    """Yield the image at *path* as a sliceable array (an `h5py.Dataset` for HDF5)."""
    path = Path(path)
    if path.suffix.lower() in HDF5_EXTENSIONS:
        f, data = open_hdf5(path, dataset)
        try:
            yield data
        finally:
            f.close()
    else:
        if dataset is not None:
            raise ValueError("dataset only applies to an HDF5 file")
        yield load_image(path)


def load_image(path: Path) -> np.ndarray:
    """Load a non-HDF5 image file into memory as a 2D or 3D array."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".mhd":
        arr = np.asarray(itk.imread(str(path)))
    elif suffix in (".tif", ".tiff"):
        arr = np.asarray(tifffile.imread(str(path)))
    else:
        with Image.open(path) as im:
            arr = np.asarray(im)

    if arr.ndim == 0:
        raise ValueError("Scalar input is not supported")

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = np.max(arr[..., :3], axis=-1)

    if arr.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape={arr.shape}")

    return arr


def open_hdf5(path: Path, dataset: str | None = None):
    """Open *path* read-only and return ``(file, dataset)``, failing early on bad input."""
    import h5py

    try:
        # Registers Blosc, LZ4, Zstd, ... with HDF5 when installed.
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass

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
