import json
import time
from pathlib import Path

from documentation_sage.evaluation.metrics import (
    calculate_average,
    recall_at_k,
    reciprocal_rank,
)
from documentation_sage.rerankers.cross_encoder import CrossEncoderReranker
from documentation_sage.retrievers.hybrid import HybridRetriever


class RAGEvaluator:

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: CrossEncoderReranker | None = None,
    ):
        self.retriever = retriever
        self.reranker = reranker

    def load_questions(
        self,
        path: Path,
    ) -> list[dict]:
        """
        Load evaluation questions from JSON.
        """

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def evaluate(
        self,
        questions: list[dict],
        top_k: int = 10,
    ) -> dict:
        """
        Evaluate retrieval performance.
        """

        recall_scores = []
        reciprocal_ranks = []
        retrieval_times = []

        print("\nRunning RAG evaluation...\n")

        for index, item in enumerate(
            questions,
            start=1,
        ):
            question = item["question"]

            expected_sources = item["expected_sources"]

            start_time = time.perf_counter()

            if self.reranker:
                candidate_k = max(top_k * 3, 30)
                candidates = self.retriever.retrieve(
                    query=question,
                    top_k=candidate_k,
                )
                results = self.reranker.rerank(
                    query=question,
                    chunks=candidates,
                    top_k=top_k,
                    score_threshold=-100.0,
                )
            else:
                results = self.retriever.retrieve(
                    query=question,
                    top_k=top_k,
                )

            end_time = time.perf_counter()

            retrieval_time = end_time - start_time

            retrieval_times.append(retrieval_time)

            retrieved_sources = []

            for result in results:
                source = result.metadata.get("source_file")

                if source and source not in retrieved_sources:
                    retrieved_sources.append(source)

            recall = recall_at_k(
                retrieved_sources,
                expected_sources,
            )

            rr = reciprocal_rank(
                retrieved_sources,
                expected_sources,
            )

            recall_scores.append(recall)

            reciprocal_ranks.append(rr)

            print(f"[{index}/{len(questions)}] " f"{question}")

            print(f"Recall@{top_k}: {recall:.2f}")

            print(f"Reciprocal Rank: {rr:.3f}")

            print(f"Time: {retrieval_time:.3f}s\n")
            print(f"Expected Sources: {expected_sources}")

            print(f"Retrieved Sources: {retrieved_sources}")

        results = {
            "total_questions": len(questions),
            f"recall_at_{top_k}": (calculate_average(recall_scores)),
            "mrr": calculate_average(reciprocal_ranks),
            "average_retrieval_time": (calculate_average(retrieval_times)),
        }

        return results
