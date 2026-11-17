import gc
import weakref

import numpy as np
import pytest

from numpy_reference import cosine_reference
from researchseek import _native


@pytest.fixture
def inputs():
    return (np.array([[1, 0], [0, 1]], dtype=np.float32),
            np.array([1, 0], dtype=np.float32), np.array([10, 20], dtype=np.int64))


@pytest.mark.parametrize("seed", [0, 23, 2026])
@pytest.mark.parametrize("dimension", [1, 7, 384])
@pytest.mark.parametrize("k", [1, 9, 113, 200])
def test_seeded_numpy_reference(seed, dimension, k):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(113, dimension))
    matrix = (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)
    query = rng.normal(size=dimension)
    query = (query / np.linalg.norm(query)).astype(np.float32)
    ids = rng.permutation(np.arange(-56, 57, dtype=np.int64))
    actual_ids, actual_scores = _native.top_k(matrix, query, ids, k)
    reference_ids, reference_scores = cosine_reference(matrix, query, ids, k)
    np.testing.assert_array_equal(actual_ids, reference_ids)
    np.testing.assert_allclose(actual_scores, reference_scores, atol=1e-5, rtol=0)


def test_identical_orthogonal_opposite_and_negative():
    matrix = np.array([[1, 0], [0, 1], [-1, 0], [-1, -1]], dtype=np.float32)
    query = np.array([1, 0], dtype=np.float32)
    ids = np.array([9, 3, 7, 11], dtype=np.int64)
    actual_ids, scores = _native.top_k(matrix, query, ids, 4)
    np.testing.assert_array_equal(actual_ids, [9, 3, 11, 7])
    np.testing.assert_allclose(scores, [1, 0, -1 / np.sqrt(2), -1], atol=1e-15)
    assert scores[-2] < 0


@pytest.mark.parametrize("k", [1, 3, 5, 6])
def test_exact_ties_duplicate_vectors_and_extreme_ids(k):
    ids = np.array([np.iinfo(np.int64).max, 0, -4, np.iinfo(np.int64).min, 27], dtype=np.int64)
    matrix = np.tile(np.array([1, -1, 2], dtype=np.float32), (5, 1))
    actual, scores = _native.top_k(matrix, matrix[0], ids, k)
    np.testing.assert_array_equal(actual, np.sort(ids)[:k])
    np.testing.assert_allclose(scores, 1, atol=1e-15)


def test_near_ties_against_reference():
    matrix = np.array([[1, 0], [1, 0.0001], [1, 0.0002], [1, 0.005]], dtype=np.float32)
    query = np.array([1, 0], dtype=np.float32)
    ids = np.array([40, 30, 20, 10], dtype=np.int64)
    actual_ids, scores = _native.top_k(matrix, query, ids, 4)
    reference_ids, expected = cosine_reference(matrix, query, ids, 4)
    np.testing.assert_allclose(scores, expected, atol=1e-5, rtol=0)
    reference_scores = dict(zip(reference_ids, expected))
    for left, right in zip(actual_ids, actual_ids[1:]):
        assert reference_scores[left] >= reference_scores[right] - 1e-5
    assert expected[0] - expected[1] < 1e-5
    assert actual_ids[-1] == reference_ids[-1]
    assert expected[0] - expected[-1] > 1e-5


@pytest.mark.parametrize("magnitude", [np.nextafter(np.float32(0), np.float32(1)),
    np.float32(1e-38), np.float32(1e-20), np.float32(1e20), np.finfo(np.float32).max])
def test_extreme_finite_magnitudes(magnitude):
    matrix = np.array([[magnitude, 0], [0, magnitude], [-magnitude, 0]], dtype=np.float32)
    query = np.array([magnitude, magnitude], dtype=np.float32)
    ids = np.array([3, 1, 2], dtype=np.int64)
    actual = _native.top_k(matrix, query, ids, 3)
    expected = cosine_reference(matrix, query, ids, 3)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-15)


def test_all_negative_scores():
    matrix = np.array([[-1, 0], [-1, 1], [-1, 2]], dtype=np.float32)
    ids, scores = _native.top_k(matrix, np.array([1, 0], dtype=np.float32), np.arange(3, dtype=np.int64), 2)
    np.testing.assert_array_equal(ids, [2, 1])
    assert (scores < 0).all()


def test_empty_matrix():
    ids, scores = _native.top_k(np.empty((0, 3), dtype=np.float32),
                               np.ones(3, dtype=np.float32), np.empty(0, dtype=np.int64), 4)
    assert ids.shape == scores.shape == (0,)
    assert ids.dtype == np.int64 and scores.dtype == np.float64


@pytest.mark.parametrize("k", [0, -1, -(10 ** 100)])
def test_nonpositive_k(inputs, k):
    with pytest.raises(ValueError, match="positive"):
        _native.top_k(*inputs, k)


@pytest.mark.parametrize("k", [True, False, 1.2, np.float64(2), "2", None])
def test_invalid_k_type(inputs, k):
    with pytest.raises(TypeError, match="integer"):
        _native.top_k(*inputs, k)


def test_numpy_integer_and_huge_positive_k(inputs):
    assert len(_native.top_k(*inputs, np.int64(1))[0]) == 1
    assert len(_native.top_k(*inputs, 10 ** 100)[0]) == 2


@pytest.mark.parametrize("index,shape", [(0, (4,)), (0, (2, 0)), (0, (1, 2, 2)),
    (1, (1, 2)), (1, (3,)), (2, (2, 1)), (2, (3,))])
def test_shapes(inputs, index, shape):
    values = list(inputs)
    values[index] = np.ones(shape, dtype=np.int64 if index == 2 else np.float32)
    with pytest.raises(ValueError, match="dimension|Expected"):
        _native.top_k(*values, 1)


@pytest.mark.parametrize("index,dtype", [(0, np.float64), (0, np.int32), (1, np.float16),
    (1, np.float64), (2, np.int32), (2, np.uint64), (2, np.float64), (0, ">f4"), (2, ">i8")])
def test_wrong_dtype_is_not_converted(inputs, index, dtype):
    values = list(inputs)
    values[index] = values[index].astype(dtype)
    with pytest.raises(TypeError, match="dtype"):
        _native.top_k(*values, 1)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_lists_are_not_converted(inputs, index):
    values = list(inputs)
    values[index] = values[index].tolist()
    with pytest.raises(TypeError):
        _native.top_k(*values, 1)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_noncontiguous_arrays(inputs, index):
    values = list(inputs)
    values[index] = np.asfortranarray(values[index]) if index == 0 else values[index][::-1]
    with pytest.raises(ValueError, match="C-contiguous"):
        _native.top_k(*values, 1)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_misaligned_arrays(inputs, index):
    values = list(inputs)
    original = values[index]
    array = np.ndarray(original.shape, dtype=original.dtype, buffer=bytearray(original.nbytes + 1), offset=1)
    array[:] = original
    values[index] = array
    with pytest.raises(ValueError, match="aligned"):
        _native.top_k(*values, 1)


@pytest.mark.parametrize("index,value", [(0, np.nan), (0, np.inf), (0, -np.inf), (1, np.nan), (1, np.inf)])
def test_nonfinite_values(inputs, index, value):
    values = list(inputs)
    values[index].flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        _native.top_k(*values, 1)


@pytest.mark.parametrize("index", [0, 1])
def test_zero_norm(inputs, index):
    values = list(inputs)
    if index == 0:
        values[index][0] = 0
    else:
        values[index][:] = 0
    with pytest.raises(ValueError, match="nonzero"):
        _native.top_k(*values, 1)


def test_duplicate_ids(inputs):
    inputs[2][:] = 10
    with pytest.raises(ValueError, match="unique"):
        _native.top_k(*inputs, 1)


def test_repeated_calls_readonly_inputs_and_owned_outputs(inputs):
    copies = [array.copy() for array in inputs]
    for array in inputs:
        array.setflags(write=False)
    first = _native.top_k(*inputs, 2)
    for _ in range(30):
        result = _native.top_k(*inputs, 2)
        for current, expected in zip(result, first):
            np.testing.assert_array_equal(current, expected)
            assert current.flags.owndata
            assert not np.shares_memory(current, expected)
    for original, copy in zip(inputs, copies):
        np.testing.assert_array_equal(original, copy)


def test_input_owners_not_retained_and_outputs_survive():
    def invoke():
        arrays = (np.eye(2, dtype=np.float32), np.ones(2, dtype=np.float32), np.array([9, 3], dtype=np.int64))
        references = [weakref.ref(array) for array in arrays]
        return _native.top_k(*arrays, 2), references
    result, references = invoke()
    gc.collect()
    assert all(reference() is None for reference in references)
    np.testing.assert_array_equal(result[0], [3, 9])
    np.testing.assert_allclose(result[1], 1 / np.sqrt(2))
