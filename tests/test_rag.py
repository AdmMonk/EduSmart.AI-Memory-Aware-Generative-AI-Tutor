"""Tests for RAG chain assembly."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.chains.rag_chain import build_rag_chain, format_docs


def test_format_docs():
    docs = [
        Document(page_content="Test content", metadata={"subject": "physics", "source_file": "physics.txt"}),
    ]
    result = format_docs(docs)
    assert "physics" in result
    assert "Test content" in result


def test_format_docs_empty():
    assert "No relevant" in format_docs([])


@patch("src.chains.rag_chain.get_llm")
def test_build_rag_chain(mock_get_llm):
    mock_llm = MagicMock()
    mock_llm.__or__ = MagicMock(return_value=MagicMock())
    mock_get_llm.return_value = mock_llm

    retriever = MagicMock()
    retriever.__or__ = MagicMock(return_value=MagicMock())

    chain = build_rag_chain(retriever)
    assert chain is not None
