from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import Settings, get_settings
from src.embeddings.embeddings import get_embeddings


def build_vectorstore(
    documents: list[Document],
    settings: Settings | None = None,
) -> FAISS:
    cfg = settings or get_settings()
    embeddings = get_embeddings(cfg)
    return FAISS.from_documents(documents, embeddings)


def save_vectorstore(
    vectorstore: FAISS,
    path: Path | None = None,
    settings: Settings | None = None,
) -> Path:
    cfg = settings or get_settings()
    target = path or cfg.vectorstore_dir
    assert target is not None
    target.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(target))
    return target


def load_vectorstore(
    path: Path | None = None,
    settings: Settings | None = None,
) -> FAISS:
    cfg = settings or get_settings()
    target = path or cfg.vectorstore_dir
    assert target is not None
    embeddings = get_embeddings(cfg)
    return FAISS.load_local(
        str(target),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(
    vectorstore: FAISS | None = None,
    settings: Settings | None = None,
) -> VectorStoreRetriever:
    cfg = settings or get_settings()
    store = vectorstore or load_vectorstore(settings=cfg)
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": cfg.retrieval_k},
    )


def index_exists(settings: Settings | None = None) -> bool:
    cfg = settings or get_settings()
    target = cfg.vectorstore_dir
    assert target is not None
    return (target / "index.faiss").exists() and (target / "index.pkl").exists()
