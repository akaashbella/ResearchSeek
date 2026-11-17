import hashlib

import numpy as np
import pytest

from researchseek.chunking import WhitespaceTokenizer
from researchseek.embeddings import normalize_embeddings


class FakeEmbedder:
    model_id = "test-only/fake"
    model_revision = "a" * 40
    dimension = 3

    def __init__(self):
        self.tokenizer = WhitespaceTokenizer()
        self.package_versions = {"test-embedder": "1"}
        self.calls = []

    def encode(self, texts, *, batch_size=32):
        self.calls.append((list(texts), batch_size))
        values = [[value + 1 for value in hashlib.sha256(text.encode()).digest()[:self.dimension]]
                  for text in texts]
        return normalize_embeddings(np.asarray(values).reshape(len(texts), self.dimension), self.dimension)


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def pytest_addoption(parser):
    parser.addoption("--real-model", action="store_true", help="Run cached real-model integration tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--real-model"):
        for item in items:
            if "real_model" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use --real-model after explicit model provisioning"))
