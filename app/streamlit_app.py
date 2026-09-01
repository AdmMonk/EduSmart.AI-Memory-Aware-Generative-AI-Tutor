import os
import sys
from pathlib import Path

import streamlit as st

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chains.conversational_chain import build_conversational_chain, invoke_tutor
from src.config import get_settings
from src.memory.conversation import memory_store
from src.retrieval.retriever import get_retriever, index_exists

st.set_page_config(
    page_title="EduSmart AI Tutor",
    page_icon="📚",
    layout="wide",
)

settings = get_settings()
settings.configure_observability()


@st.cache_resource
def load_chain():
    if not index_exists(settings):
        return None
    retriever = get_retriever(settings=settings)
    return build_conversational_chain(retriever)


def init_session_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = memory_store.create_session()
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    init_session_state()
    chain = load_chain()

    st.title("📚 EduSmart AI Tutor")
    st.caption(
        "Personalized, memory-aware tutoring powered by LangChain RAG + Hugging Face"
    )

    with st.sidebar:
        st.header("Settings")
        st.write(f"**Model:** `{settings.llm_model}`")
        st.write(f"**Embeddings:** `{settings.embedding_model}`")
        st.write(f"**Index ready:** {'✅' if index_exists() else '❌'}")

        if not index_exists():
            st.warning(
                "Vector index not found. Run:\n\n"
                "`python scripts/build_index.py`"
            )

        if st.button("New session"):
            st.session_state.session_id = memory_store.create_session()
            st.session_state.messages = []
            st.rerun()

        if st.button("Clear history"):
            memory_store.clear_session(st.session_state.session_id)
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.markdown(
            "**Privacy:** Sessions expire after "
            f"{settings.session_retention_hours}h. "
            "No personal data is collected."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask your tutor a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if chain is None:
                st.error("Please build the vector index first.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        answer = invoke_tutor(
                            chain, prompt, st.session_state.session_id
                        )
                    except Exception as exc:
                        answer = (
                            f"Sorry, I encountered an error. "
                            f"Ensure HF_TOKEN is set in `.env`.\n\n`{exc}`"
                        )
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )


if __name__ == "__main__":
    main()
