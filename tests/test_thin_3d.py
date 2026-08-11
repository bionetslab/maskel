"""Unit tests for the 3D thinning kernel's scratch-buffer bounds.

These cover the assumptions the reduced-size scratch buffers rely on:
``candidates``/``removable`` are sized to the initial foreground count, and
``removed_epoch`` is a uint8 stamp whose tag is recycled every 255 batches.
"""

import numpy as np
import pytest
from skimage.morphology import skeletonize

from vesskel.thin_3d import _MAX_DIM, thin_3d


def _cube_with_rod(n: int, rod_len: int) -> np.ndarray:
    """Solid n-cube plus a 1-voxel rod sticking out of one face.

    The solid cube needs many peeling passes (which is what drives the tag
    count up), while the rod is protected by the endpoint criterion and so
    survives into the skeleton - giving a non-trivial result to compare.
    """
    vol = np.zeros((n, n, n + rod_len), dtype=np.uint8)
    vol[:, :, :n] = 1
    vol[n // 2, n // 2, n : n + rod_len] = 1
    return vol


class TestExtentGuard:
    def test_rejects_axis_longer_than_uint16(self):
        # Coordinates are stored as uint16, so a longer axis would silently
        # wrap inside the numba kernels - it has to be refused up front.
        oversized = np.zeros((1, 1, _MAX_DIM + 1), dtype=np.uint8)

        with pytest.raises(ValueError, match="Volume too large"):
            thin_3d(oversized)

    def test_accepts_axis_exactly_at_the_limit(self):
        # _MAX_DIM itself must still be allowed (the bound is inclusive).
        at_limit = np.zeros((1, 1, _MAX_DIM), dtype=np.uint8)
        at_limit[0, 0, 10:20] = 1

        result = thin_3d(at_limit)

        assert result.shape == at_limit.shape
        # A 1-voxel-thick line is all endpoints, so nothing may be removed.
        assert np.array_equal(result, at_limit)

    def test_dimension_error_mentions_the_shape(self):
        oversized = np.zeros((1, _MAX_DIM + 1, 1), dtype=np.uint8)

        with pytest.raises(ValueError, match=str(_MAX_DIM)):
            thin_3d(oversized)


class TestForegroundSizedBuffers:
    def test_empty_volume_needs_no_candidate_slots(self):
        # Foreground count 0 -> zero-length scratch buffers; the sweep loop
        # must find no candidates rather than index into them.
        vol = np.zeros((8, 8, 8), dtype=np.uint8)

        result = thin_3d(vol)

        assert not result.any()
        assert result.shape == vol.shape

    def test_fully_solid_volume(self):
        # Every voxel is foreground, so the buffers are sized to the whole
        # volume - the upper-bound case for the foreground-count sizing.
        vol = np.ones((12, 12, 12), dtype=np.uint8)

        result = thin_3d(vol)

        assert np.array_equal(result.astype(bool), skeletonize(vol))

    def test_sparse_volume(self):
        # Foreground is a tiny fraction of the volume, which is where the
        # foreground-count sizing pays off.
        vol = np.zeros((40, 40, 40), dtype=np.uint8)
        vol[20, 20, 5:35] = 1
        vol[20, 5:35, 20] = 1
        vol[5:35, 20, 20] = 1

        result = thin_3d(vol)

        assert np.array_equal(result.astype(bool), skeletonize(vol))


class TestRemovedEpochTagWrap:
    def test_volume_needing_more_than_255_batches(self):
        # This volume takes ~276 removal batches, so the uint8 tag wraps and
        # removed_epoch gets wiped mid-run.  If the wipe or the recycling were
        # wrong, stale stamps would alias the live tag and voxels would be
        # removed without the simplicity re-check they need.
        vol = _cube_with_rod(90, 20)

        result = thin_3d(vol)

        assert np.array_equal(result.astype(bool), skeletonize(vol))
        assert result.any(), "the rod must survive as skeleton"
