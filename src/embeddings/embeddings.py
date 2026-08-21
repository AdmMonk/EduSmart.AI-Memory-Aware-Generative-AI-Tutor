from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Settings, get_settings


def get_embeddings(settings: Settings | None = None) -> HuggingFaceEmbeddings:
    cfg = settings or get_settings()
    return HuggingFaceEmbeddings(
        model_name=cfg.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
