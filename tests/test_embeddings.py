import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from researchseek.chunking import ChunkConfig, chunk_text
from researchseek.embeddings import (
    DEFAULT_MODEL, DEFAULT_REVISION, EmbeddingError, ModelTokenizer,
    SentenceTransformerEmbedder, normalize_embeddings, validate_embeddings,
    vector_from_blob, vector_to_blob,
)


def test_normalization_and_float32():
    result = normalize_embeddings([[3, 4], [0, -7]], 2, rows=2)
    assert result.dtype == np.float32 and result.flags.c_contiguous
    np.testing.assert_allclose(result, [[0.6, 0.8], [0, -1]], atol=1e-7)
    assert normalize_embeddings(np.empty((0, 2)), 2).shape == (0, 2)


@pytest.mark.parametrize("values", [[[0, 0]], [[np.nan, 1]], [[np.inf, 1]], [[-np.inf, 1]],
                                  [[1, 2, 3]], [1, 2], [["one", "two"]], [[1j, 1]]])
def test_invalid_embeddings(values):
    with pytest.raises(ValueError):
        normalize_embeddings(values, 2, rows=1)


@pytest.mark.parametrize("values", [[[0, 0]], [[0.6, 0.7]], [[np.nan, 0]], [[np.inf, 0]]])
def test_storage_validation_requires_unit_vectors(values):
    with pytest.raises(ValueError):
        validate_embeddings(values, 2)


def test_wrong_rows():
    with pytest.raises(ValueError, match="matrix"):
        normalize_embeddings([[1, 0]], 2, rows=2)


def test_little_endian_roundtrip():
    blob = vector_to_blob(np.array([0.6, 0.8], dtype=">f4"), 2)
    assert blob == struct.pack("<ff", 0.6, 0.8)
    np.testing.assert_allclose(vector_from_blob(blob, 2), [0.6, 0.8])


@pytest.mark.parametrize("blob", [b"", b"abcd", struct.pack("<ff", 0, 0), struct.pack("<ff", np.nan, 0)])
def test_invalid_blob(blob):
    with pytest.raises(ValueError):
        vector_from_blob(blob, 2)


class BackendTokenizer:
    is_fast = True
    model_max_length = 8
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{"tokenizer":"stub-fast"}')

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, *, add_special_tokens, truncation, return_offsets_mapping=False, verbose=False):
        assert not truncation
        spans = [(index, index + 1) for index, value in enumerate(text) if not value.isspace()]
        return {"input_ids": [1] * (len(spans) + (2 if add_special_tokens else 0)), "offset_mapping": spans}


@pytest.fixture
def mocked_model(monkeypatch):
    calls = {"snapshot": [], "load": [], "encode": []}

    def download(**kwargs):
        calls["snapshot"].append(kwargs)
        return f"/cache/snapshots/{DEFAULT_REVISION}"

    class Model:
        tokenizer = BackendTokenizer()
        max_seq_length = 7

        def __init__(self, path, **kwargs):
            calls["load"].append((path, kwargs))

        def get_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            calls["encode"].append((texts, kwargs))
            return np.array([[3, 4]] * len(texts))

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model))
    return calls


def test_offline_identity_loading_and_encoding(mocked_model):
    model = SentenceTransformerEmbedder(revision="release-tag")
    assert model.model_id == DEFAULT_MODEL
    assert model.model_revision == DEFAULT_REVISION
    assert mocked_model["snapshot"][0]["local_files_only"] is True
    assert mocked_model["load"][0][1] == {"device": "cpu", "local_files_only": True, "trust_remote_code": False}
    np.testing.assert_allclose(model.encode(["abc"], batch_size=2), [[0.6, 0.8]])
    assert mocked_model["encode"][0][1]["batch_size"] == 2
    assert mocked_model["encode"][0][1]["prompt"] == ""
    assert model.encode([]).shape == (0, 2)


def test_download_requires_explicit_flag(mocked_model):
    SentenceTransformerEmbedder(allow_download=True)
    assert mocked_model["snapshot"][0]["local_files_only"] is False


def test_requested_commit_must_match(mocked_model):
    with pytest.raises(EmbeddingError, match="revision does not match"):
        SentenceTransformerEmbedder(revision="b" * 40)


def test_mutable_cache_path_rejected(mocked_model, monkeypatch):
    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", lambda **kwargs: "/cache/main")
    with pytest.raises(EmbeddingError, match="immutable"):
        SentenceTransformerEmbedder()


def test_missing_artifacts_actionable(mocked_model, monkeypatch):
    def unavailable(**kwargs):
        raise OSError("missing weights")
    monkeypatch.setattr(sys.modules["huggingface_hub"], "snapshot_download", unavailable)
    with pytest.raises(EmbeddingError, match="--download-model"):
        SentenceTransformerEmbedder()


@pytest.mark.parametrize("texts", [["abcdef"], [""], "abc"])
def test_refuse_truncation_and_invalid_input(mocked_model, texts):
    model = SentenceTransformerEmbedder()
    with pytest.raises(ValueError):
        model.encode(texts)
    assert mocked_model["encode"] == []


def test_model_tokenizer_limits_and_offsets(mocked_model):
    tokenizer = ModelTokenizer(BackendTokenizer(), DEFAULT_MODEL, DEFAULT_REVISION, 7)
    assert tokenizer.max_tokens == 7 and tokenizer.special_tokens == 2
    assert DEFAULT_REVISION in tokenizer.identity
    chunks = chunk_text("αβ γδ\n\nεζηθικ", tokenizer, ChunkConfig(7, 1))
    assert chunks == chunk_text("αβ γδ\n\nεζηθικ", tokenizer, ChunkConfig(7, 1))
    assert all(tokenizer.count_tokens(chunk.text) <= 7 for chunk in chunks)


def test_slow_tokenizer_rejected():
    tokenizer = BackendTokenizer()
    tokenizer.is_fast = False
    with pytest.raises(EmbeddingError, match="fast tokenizer"):
        ModelTokenizer(tokenizer, DEFAULT_MODEL, DEFAULT_REVISION, 7)
