"""Lee94 thinning algorithm dispatcher.

This module dispatches to 2D or 3D thinning implementations based on input dimension.
"""


def lee94_thin(img):
    """Thin a binary image/volume.

    Parameters
    ----------
    img : ndarray
        2D binary image or 3D binary volume (0=background, 1=foreground).

    Returns
    -------
    ndarray
        Thinned image/volume with the same shape as the input.
    """
    if img.ndim == 2:
        from .thin_2d import thin_2d as _impl

    elif img.ndim == 3:
        from .thin_3d import thin_3d as _impl
    else:
        raise ValueError(f"Expected 2D or 3D input, got {img.ndim}D")
    return _impl(img)
