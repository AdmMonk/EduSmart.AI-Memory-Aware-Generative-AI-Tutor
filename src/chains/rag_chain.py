from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever

from src.llm.model import get_llm
from src.prompts.tutor_prompt import QA_PROMPT


def format_docs(docs: list) -> str:
    if not docs:
        return "No relevant curriculum content found."
    parts = []
    for i, doc in enumerate(docs, 1):
        subject = doc.metadata.get("subject", "unknown")
        source = doc.metadata.get("source_file", doc.metadata.get("source", ""))
        parts.append(f"[{i}] Subject: {subject} | Source: {source}\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(retriever: VectorStoreRetriever):
    llm = get_llm()

    rag_chain = (
        RunnableMap(
            {
                "context": itemgetter("input") | retriever | format_docs,
                "input": itemgetter("input"),
                "chat_history": itemgetter("chat_history"),
            }
        )
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain


def build_simple_rag_chain(retriever: VectorStoreRetriever):
    llm = get_llm()

    chain = (
        {
            "context": RunnablePassthrough() | retriever | format_docs,
            "input": RunnablePassthrough(),
            "chat_history": lambda _: [],
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain
