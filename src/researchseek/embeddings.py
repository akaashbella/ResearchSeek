"""CPU embeddings, immutable model identity, and vector validation."""

import hashlib
from importlib.metadata import version
from pathlib import Path
import re
from typing import Protocol, Sequence

import numpy as np

from researchseek.chunking import Tokenizer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_CACHE = Path(".tools/model-cache")
NORM_TOLERANCE = 1e-5


class EmbeddingError(RuntimeError):
    """Model artifacts or encoding are unavailable."""


class Embedder(Protocol):
    model_id: str
    model_revision: str
    dimension: int
    tokenizer: Tokenizer
    package_versions: dict[str, str]

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        ...


def _matrix(values: object, dimension: int, rows: int | None) -> np.ndarray:
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Embedding dimension must be a positive integer")
    matrix = np.asarray(values)
    if matrix.dtype.kind not in "fiu":
        raise ValueError("Embeddings must contain real numeric values")
    matrix = matrix.astype(np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != dimension or (rows is not None and matrix.shape[0] != rows):
        raise ValueError(f"Expected embedding matrix ({rows if rows is not None else 'N'}, {dimension})")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain NaN or infinity")
    return matrix


def normalize_embeddings(values: object, dimension: int, *, rows: int | None = None) -> np.ndarray:
    matrix = _matrix(values, dimension, rows)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Embedding norms must be finite and nonzero")
    normalized = np.ascontiguousarray(matrix / norms[:, None], dtype=np.float32)
    return validate_embeddings(normalized, dimension, rows=rows)


def validate_embeddings(values: object, dimension: int, *, rows: int | None = None) -> np.ndarray:
    matrix = _matrix(values, dimension, rows)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=NORM_TOLERANCE, rtol=0):
        raise ValueError("Embeddings must have unit L2 norm (tolerance 1e-5)")
    result = np.ascontiguousarray(matrix, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Embeddings cannot be represented as finite float32 values")
    return result


def vector_to_blob(vector: object, dimension: int) -> bytes:
    array = np.asarray(vector)
    if array.ndim != 1:
        raise ValueError("Expected a single embedding vector")
    matrix = validate_embeddings(array[None, :], dimension, rows=1)
    return matrix.astype("<f4").tobytes()


def vector_from_blob(blob: bytes, dimension: int) -> np.ndarray:
    if len(blob) != 4 * dimension:
        raise ValueError("Embedding byte length does not match configured dimension")
    array = np.frombuffer(blob, dtype="<f4")
    return validate_embeddings(array[None, :], dimension, rows=1)[0].copy()


class ModelTokenizer:
    def __init__(self, tokenizer: object, model_id: str, revision: str, max_tokens: int):
        if not tokenizer.is_fast:
            raise EmbeddingError("A fast tokenizer with source offsets is required")
        self._tokenizer = tokenizer
        self.max_tokens = min(max_tokens, tokenizer.model_max_length)
        self.special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        digest = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        self.identity = f"{model_id}@{revision}:fast-offsets-v1:{digest}"

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tokenizer(text, add_special_tokens=False, truncation=False,
                                  return_offsets_mapping=True, verbose=False)
        return [tuple(span) for span in encoded["offset_mapping"]]

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=True, truncation=False, verbose=False)["input_ids"])


class SentenceTransformerEmbedder:
    def __init__(self, model_id: str = DEFAULT_MODEL, *, revision: str = DEFAULT_REVISION,
                 cache_dir: Path = DEFAULT_CACHE, allow_download: bool = False):
        try:
            from huggingface_hub import snapshot_download
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingError("Embedding dependencies are unavailable; reinstall with pip install -e '.[dev]'") from error

        try:
            snapshot = Path(snapshot_download(
                repo_id=model_id, revision=revision, cache_dir=str(cache_dir),
                local_files_only=not allow_download,
                allow_patterns=["*.json", "*.safetensors", "vocab.txt", "tokenizer.model", "LICENSE*", "README.md"],
                token=False,
            ))
            resolved = snapshot.name
            if not re.fullmatch(r"[0-9a-f]{40}", resolved):
                raise EmbeddingError("Model cache did not resolve to an immutable commit SHA")
            if re.fullmatch(r"[0-9a-f]{40}", revision) and resolved != revision:
                raise EmbeddingError("Resolved model revision does not match requested commit")
            self._model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True,
                                              trust_remote_code=False)
        except Exception as error:
            raise EmbeddingError(
                f"Cannot load {model_id}@{revision} from {cache_dir}. "
                "Provision the cache with 'index ... --download-model' while online, "
                f"then retry without that flag. Details: {error}"
            ) from error
        self.model_id = model_id
        self.model_revision = resolved
        dimension_method = getattr(self._model, "get_embedding_dimension", None)
        self.dimension = (dimension_method or self._model.get_sentence_embedding_dimension)()
        if type(self.dimension) is not int or self.dimension < 1:
            raise EmbeddingError("Model did not report a valid embedding dimension")
        self.tokenizer = ModelTokenizer(self._model.tokenizer, model_id, resolved, self._model.max_seq_length)
        self.package_versions = {name: version(name) for name in (
            "sentence-transformers", "transformers", "tokenizers", "torch", "numpy", "huggingface-hub")}

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Expected a sequence of nonempty passage/query strings")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Batch size must be a positive integer")
        if any(self.tokenizer.count_tokens(text) > self.tokenizer.max_tokens for text in texts):
            raise ValueError("Input exceeds the model token limit; refusing silent truncation")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        try:
            values = self._model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                                        normalize_embeddings=False, show_progress_bar=False, prompt="")
            return normalize_embeddings(values, self.dimension, rows=len(texts))
        except Exception as error:
            raise EmbeddingError(f"CPU embedding failed: {error}") from error
