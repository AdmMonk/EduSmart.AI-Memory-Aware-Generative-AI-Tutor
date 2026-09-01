from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Callable
from src.chains.rag_chain import format_docs
from src.llm.model import get_llm
from src.memory.conversation import memory_store
from src.prompts.tutor_prompt import CONDENSE_QUESTION_PROMPT, QA_PROMPT


def build_conversational_chain(
    retriever: VectorStoreRetriever,
    get_session_history: Callable | None = None,
):
    llm = get_llm()

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        CONDENSE_QUESTION_PROMPT,
    )

    document_chain = create_stuff_documents_chain(llm, QA_PROMPT)

    # Override context formatting via a wrapper
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    session_fn = get_session_history or memory_store.get_history

    conversational_chain = RunnableWithMessageHistory(
        retrieval_chain,
        session_fn,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_chain


def invoke_tutor(
    chain,
    question: str,
    session_id: str,
) -> str:
    result = chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": session_id}},
    )
    return result.get("answer", result.get("output", str(result)))
