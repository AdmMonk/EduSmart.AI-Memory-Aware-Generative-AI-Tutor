import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.chains.conversational_chain import build_conversational_chain, invoke_tutor
from src.config import get_settings
from src.memory.conversation import memory_store
from src.retrieval.retriever import get_retriever, index_exists

logger = logging.getLogger(__name__)

_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain
    settings = get_settings()
    settings.configure_observability()

    if index_exists(settings):
        retriever = get_retriever(settings=settings)
        _chain = build_conversational_chain(retriever)
        logger.info("Conversational RAG chain initialized.")
    else:
        logger.warning("Vector index not found. Run: python scripts/build_index.py")

    yield

    _chain = None


app = FastAPI(
    title="EduSmart AI Tutor API",
    description="RAG-powered personalized tutoring with LangChain + Hugging Face",
    version="0.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class SessionResponse(BaseModel):
    session_id: str


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "index_ready": index_exists(),
        "chain_ready": _chain is not None,
    }


@app.post("/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    """Create an anonymous tutoring session (privacy-first)."""
    session_id = memory_store.create_session()
    return SessionResponse(session_id=session_id)


@app.delete("/sessions/{session_id}")
def clear_session(session_id: str) -> dict[str, str]:
    """Clear conversation history for a session."""
    memory_store.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Send a message to the AI tutor."""
    if _chain is None:
        raise HTTPException(
            status_code=503,
            detail="Vector index not built. Run: python scripts/build_index.py",
        )

    settings = get_settings()
    session_id = request.session_id or memory_store.create_session()

    if settings.log_queries:
        logger.info("Query session=%s", session_id)

    try:
        answer = invoke_tutor(_chain, request.message, session_id)
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(answer=answer, session_id=session_id)


@app.get("/evaluation/retrieval")
def evaluation_retrieval() -> dict:
    """Run lightweight retrieval metrics on the default eval set."""
    from src.evaluation.evaluate import run_evaluation

    if not index_exists():
        raise HTTPException(status_code=503, detail="Index not built")
    retriever = get_retriever()
    return run_evaluation(retriever)
