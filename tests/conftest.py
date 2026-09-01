"""Shared pytest fixtures."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_data_dir() -> Path:
    return ROOT / "data" / "raw"


@pytest.fixture
def sample_chunks() -> list[Document]:
    return [
        Document(
            page_content="Newton's first law: inertia keeps objects in motion.",
            metadata={"subject": "physics", "source_file": "physics.txt"},
        ),
        Document(
            page_content="Quadratic formula: x = (-b ± sqrt(b²-4ac)) / 2a",
            metadata={"subject": "mathematics", "source_file": "mathematics.txt"},
        ),
        Document(
            page_content="WWI caused by alliance system and assassination in Sarajevo.",
            metadata={"subject": "history", "source_file": "history.txt"},
        ),
    ]
