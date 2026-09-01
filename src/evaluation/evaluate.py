from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


@dataclass
class RetrievalMetrics:

    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    num_queries: int = 0


@dataclass
class EvaluationSample:

    question: str
    expected_keywords: list[str] = field(default_factory=list)
    subject: str | None = None


def _doc_matches(doc: Document, keywords: list[str]) -> bool:
    content = doc.page_content.lower()
    return any(kw.lower() in content for kw in keywords)


def evaluate_retrieval(
    retriever: VectorStoreRetriever,
    samples: list[EvaluationSample],
    k: int = 4,
) -> RetrievalMetrics:
    if not samples:
        return RetrievalMetrics()

    hits = 0
    precisions: list[float] = []
    reciprocal_ranks: list[float] = []

    for sample in samples:
        docs = retriever.invoke(sample.question)
        if len(docs) > k:
            docs = docs[:k]

        relevant = [_doc_matches(d, sample.expected_keywords) for d in docs]
        num_relevant = sum(relevant)
        precisions.append(num_relevant / max(len(docs), 1))

        if num_relevant > 0:
            hits += 1
            for rank, is_rel in enumerate(relevant, 1):
                if is_rel:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(0.0)

    n = len(samples)
    return RetrievalMetrics(
        precision_at_k=sum(precisions) / n,
        recall_at_k=hits / n,
        mrr=sum(reciprocal_ranks) / n,
        hit_rate=hits / n,
        num_queries=n,
    )


def default_eval_samples() -> list[EvaluationSample]:
    return [
        EvaluationSample(
            question="What is Newton's first law of motion?",
            expected_keywords=["inertia", "Newton", "motion"],
            subject="physics",
        ),
        EvaluationSample(
            question="How do you solve a quadratic equation?",
            expected_keywords=["quadratic", "formula", "equation"],
            subject="mathematics",
        ),
        EvaluationSample(
            question="What caused World War I?",
            expected_keywords=["alliance", "assassination", "war"],
            subject="history",
        ),
    ]


def run_evaluation(retriever: VectorStoreRetriever) -> dict:
    metrics = evaluate_retrieval(retriever, default_eval_samples())
    return {
        "precision_at_k": round(metrics.precision_at_k, 3),
        "recall_at_k": round(metrics.recall_at_k, 3),
        "mrr": round(metrics.mrr, 3),
        "hit_rate": round(metrics.hit_rate, 3),
        "num_queries": metrics.num_queries,
    }
