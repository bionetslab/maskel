import numpy as np

from maskel._utils import to_binary


class TestToBinary:
    def test_output_dtype(self):
        result = to_binary(np.array([0, 1, 2, 3], dtype=np.int64))
        assert result.dtype == np.uint8

    def test_zero_stays_zero(self):
        result = to_binary(np.array([0, 0, 0], dtype=np.uint8))
        assert np.array_equal(result, [0, 0, 0])

    def test_nonzero_becomes_one(self):
        result = to_binary(np.array([0, 1, 5, 255], dtype=np.int32))
        assert np.array_equal(result, [0, 1, 1, 1])

    def test_boolean_input(self):
        result = to_binary(np.array([False, True, False, True]))
        assert np.array_equal(result, [0, 1, 0, 1])

    def test_float_input(self):
        result = to_binary(np.array([0.0, 0.5, 1e-10, 2.718], dtype=np.float64))
        assert np.array_equal(result, [0, 1, 1, 1])

    def test_preserves_shape(self):
        arr = np.random.default_rng(42).integers(0, 256, size=(8, 8, 8))
        result = to_binary(arr)
        assert result.shape == arr.shape

    def test_does_not_mutate_input(self):
        original = np.array([0, 2, 0, 4], dtype=np.int32)
        copy_before = original.copy()
        to_binary(original)
        assert np.array_equal(original, copy_before)

    def test_negative_floats(self):
        result = to_binary(np.array([-0.5, -1.0, -3.14]))
        assert np.array_equal(result, [0, 0, 0])

    def test_nan_and_inf(self):
        arr = np.array([np.nan, np.inf, -np.inf], dtype=np.float64)
        result = to_binary(arr)
        assert result.dtype == np.uint8

    def test_empty_array(self):
        result = to_binary(np.array([], dtype=np.int64).reshape(0, 5))
        assert result.shape == (0, 5)
        assert result.dtype == np.uint8
