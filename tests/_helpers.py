import numpy as np


def cross_image(size: int = 32) -> np.ndarray:
    """Binary cross: one junction and four endpoints."""
    img = np.zeros((size, size), dtype=np.uint8)
    img[size // 2, size // 4 : 3 * size // 4] = 1
    img[size // 4 : 3 * size // 4, size // 2] = 1
    return img


def loop_image(size: int = 20) -> np.ndarray:
    """Binary rectangle ring: two junctions connected by parallel branches."""
    img = np.zeros((size, size), dtype=np.uint8)
    margin = size // 4
    inner = size - margin
    img[margin, margin:inner] = 1
    img[inner - 1, margin:inner] = 1
    img[margin:inner, margin] = 1
    img[margin:inner, inner - 1] = 1
    return img


def cross_volume(size: int = 16) -> np.ndarray:
    """Binary volume with two perpendicular lines crossing at the center."""
    vol = np.zeros((size, size, size), dtype=np.uint8)
    vol[size // 2, size // 2, :] = 1
    vol[size // 2, :, size // 2] = 1
    return vol


def line_volume(shape: tuple[int, int, int], axis: int = 0) -> np.ndarray:
    """Binary volume with a single straight line through the center along *axis*."""
    vol = np.zeros(shape, dtype=np.uint8)
    idx = [s // 2 for s in shape]
    idx[axis] = slice(None)
    vol[tuple(idx)] = 1
    return vol
