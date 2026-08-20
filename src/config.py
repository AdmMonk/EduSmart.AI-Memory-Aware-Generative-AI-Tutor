"""Application configuration loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the AI tutor system."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    data_raw_dir: Path | None = None
    data_processed_dir: Path | None = None
    vectorstore_dir: Path | None = None

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    use_hf_inference_api: bool = True

    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_k: int = 4
    memory_window: int = 10

    # Privacy & data minimization
    session_retention_hours: int = 24
    log_queries: bool = False

    # Optional LangSmith observability
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = "edusmart-ai-tutor"
    langchain_tracing_v2: bool = False

    def model_post_init(self, __context: object) -> None:
        if self.data_raw_dir is None:
            self.data_raw_dir = self.project_root / "data" / "raw"
        if self.data_processed_dir is None:
            self.data_processed_dir = self.project_root / "data" / "processed"
        if self.vectorstore_dir is None:
            self.vectorstore_dir = self.project_root / "vectorstore" / "faiss_index"

    def configure_observability(self) -> None:
        """Enable LangSmith tracing when configured."""
        import os

        if self.langchain_tracing_v2 and self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project


def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
