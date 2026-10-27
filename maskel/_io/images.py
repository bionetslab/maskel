"""Loading input images."""

from __future__ import annotations

from pathlib import Path

import itk
import numpy as np
import tifffile
from PIL import Image


def load_image(path: Path) -> np.ndarray:
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
