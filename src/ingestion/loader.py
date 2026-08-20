"""Load curriculum documents and split them into chunks for embedding."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import Settings, get_settings


def load_documents(source_dir: Path | None = None) -> list[Document]:
    """Load PDF and text files from the raw data directory."""
    settings = get_settings()
    directory = source_dir or settings.data_raw_dir
    assert directory is not None

    documents: list[Document] = []
    if not directory.exists():
        return documents

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["subject"] = path.stem
                doc.metadata["source_file"] = path.name
            documents.extend(docs)
        elif path.suffix.lower() in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["subject"] = path.stem
                doc.metadata["source_file"] = path.name
            documents.extend(docs)

    return documents


def split_documents(
    documents: list[Document],
    settings: Settings | None = None,
) -> list[Document]:
    """Split documents into overlapping chunks suitable for retrieval."""
    cfg = settings or get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_and_split(source_dir: Path | None = None) -> list[Document]:
    """Load and split all curriculum documents in one step."""
    return split_documents(load_documents(source_dir))
