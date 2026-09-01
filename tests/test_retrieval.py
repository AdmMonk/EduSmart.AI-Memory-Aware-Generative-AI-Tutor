"""Tests for FAISS retrieval."""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.evaluation.evaluate import EvaluationSample, evaluate_retrieval
from src.retrieval.retriever import build_vectorstore, index_exists, save_vectorstore


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings for unit tests."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 384


@pytest.fixture
def mock_embeddings():
    with patch("src.retrieval.retriever.get_embeddings", return_value=FakeEmbeddings()):
        yield FakeEmbeddings()


def test_build_and_save_vectorstore(mock_embeddings, sample_chunks, tmp_path):
    from src.config import Settings

    settings = Settings(vectorstore_dir=tmp_path / "faiss_index")
    store = build_vectorstore(sample_chunks, settings)
    path = save_vectorstore(store, settings=settings)
    assert index_exists(settings)
    assert (path / "index.faiss").exists()


def test_retrieval_keyword_match(mock_embeddings, sample_chunks, tmp_path):
    from src.config import Settings
    from src.retrieval.retriever import get_retriever, load_vectorstore

    settings = Settings(vectorstore_dir=tmp_path / "faiss_index", retrieval_k=3)
    store = build_vectorstore(sample_chunks, settings)
    save_vectorstore(store, settings=settings)

    loaded = load_vectorstore(settings=settings)
    retriever = get_retriever(loaded, settings)

    samples = [
        EvaluationSample(
            question="Newton first law inertia",
            expected_keywords=["inertia", "Newton"],
        )
    ]
    metrics = evaluate_retrieval(retriever, samples, k=3)
    assert metrics.num_queries == 1
