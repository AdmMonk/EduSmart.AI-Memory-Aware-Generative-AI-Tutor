"""Tests for document ingestion."""

from pathlib import Path

from src.ingestion.loader import ingest_and_split, load_documents, split_documents


def test_load_sample_documents(sample_data_dir: Path):
    docs = load_documents(sample_data_dir)
    assert len(docs) >= 3
    subjects = {d.metadata.get("subject") for d in docs}
    assert "physics" in subjects
    assert "mathematics" in subjects
    assert "history" in subjects


def test_split_documents(sample_data_dir: Path):
    docs = load_documents(sample_data_dir)
    chunks = split_documents(docs)
    assert len(chunks) > len(docs)
    assert all(len(c.page_content) <= 600 for c in chunks)


def test_ingest_and_split(sample_data_dir: Path):
    chunks = ingest_and_split(sample_data_dir)
    assert len(chunks) >= 3
    assert all("subject" in c.metadata for c in chunks)
