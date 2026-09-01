from datetime import datetime, timedelta
from uuid import uuid4

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage

from src.config import Settings, get_settings


class SessionMemoryStore:

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._sessions: dict[str, ChatMessageHistory] = {}
        self._created_at: dict[str, datetime] = {}

    def create_session(self) -> str:
        session_id = str(uuid4())
        self._sessions[session_id] = ChatMessageHistory()
        self._created_at[session_id] = datetime.utcnow()
        return session_id

    def _purge_expired(self) -> None:
        retention = self._settings.session_retention_hours
        if retention <= 0:
            return
        cutoff = datetime.utcnow() - timedelta(hours=retention)
        expired = [sid for sid, ts in self._created_at.items() if ts < cutoff]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._created_at.pop(sid, None)

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        self._purge_expired()
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMessageHistory()
            self._created_at[session_id] = datetime.utcnow()
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._created_at.pop(session_id, None)

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        return self.get_history(session_id).messages


# Module-level singleton for API/Streamlit use
memory_store = SessionMemoryStore()
