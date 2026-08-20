"""Internal utility functions shared across the maskel package."""

import numpy as np


def to_binary(arr: np.ndarray, *, inplace: bool = False) -> np.ndarray:
    """Convert an array to a binary uint8 mask (0=background, >0=foreground).

    Parameters
    ----------
    arr : ndarray
        Input array. Non-zero values become 1.
    inplace : bool
        When True *and* the input is a writeable ``uint8`` array, rewrite the
        values in the caller's buffer instead of allocating a new array. This
        mutates *arr*, so only pass True when the caller owns the memory -
        never for napari layer data or anything else the user still holds a
        reference to.

        A request that cannot be honoured falls back to allocating rather than
        raising: another dtype would not fit the input buffer, and readers such
        as `PIL` (via ``np.asarray``) hand back read-only arrays. So *inplace*
        is a memory optimisation, never a guarantee about *arr*.

    Returns
    -------
    ndarray
        Binary uint8 array. The same object as *arr* when written in place.
    """
    if inplace and arr.dtype == np.uint8 and arr.flags.writeable:
        np.greater(arr, 0, out=arr, casting="unsafe")
        return arr

    # np.greater yields a bool array, which is already one byte per element
    # holding exactly 0/1 - viewing it as uint8 is free. Going through
    # `.astype(np.uint8)` instead would allocate a second full-size buffer.
    return np.greater(arr, 0).view(np.uint8)
