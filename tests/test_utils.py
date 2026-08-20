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


class TestToBinaryInplace:
    def test_uint8_is_written_in_place(self):
        arr = np.array([0, 2, 0, 255], dtype=np.uint8)
        result = to_binary(arr, inplace=True)
        assert result is arr
        assert np.array_equal(arr, [0, 1, 0, 1])

    def test_uint8_already_binary_is_unchanged(self):
        arr = np.array([0, 1, 1, 0], dtype=np.uint8)
        assert np.array_equal(to_binary(arr, inplace=True), [0, 1, 1, 0])

    def test_read_only_uint8_falls_back_to_a_copy(self):
        """Readers like PIL hand back read-only arrays; must not raise."""
        arr = np.array([0, 2, 0, 255], dtype=np.uint8)
        arr.flags.writeable = False

        result = to_binary(arr, inplace=True)

        assert result is not arr
        assert result.flags.writeable
        assert np.array_equal(result, [0, 1, 0, 1])
        assert np.array_equal(arr, [0, 2, 0, 255])

    def test_non_uint8_falls_back_to_a_copy(self):
        arr = np.array([0, 2, 0, 4], dtype=np.int32)
        result = to_binary(arr, inplace=True)
        assert result is not arr
        assert result.dtype == np.uint8
        assert np.array_equal(arr, [0, 2, 0, 4])
        assert np.array_equal(result, [0, 1, 0, 1])

    def test_multidimensional(self):
        arr = np.array([[0, 7], [3, 0]], dtype=np.uint8)
        to_binary(arr, inplace=True)
        assert np.array_equal(arr, [[0, 1], [1, 0]])

    def test_default_still_copies(self):
        arr = np.array([0, 2, 0, 255], dtype=np.uint8)
        result = to_binary(arr)
        assert result is not arr
        assert np.array_equal(arr, [0, 2, 0, 255])
