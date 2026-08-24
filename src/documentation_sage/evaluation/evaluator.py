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
        Evaluate retrieval and reranking performance.
        """

        recall_scores = []
        reciprocal_ranks = []

        retrieval_times = []
        rerank_times = []
        total_times = []

        print("\nRunning RAG evaluation...\n")

        for index, item in enumerate(
            questions,
            start=1,
        ):
            question = item["question"]
            expected_sources = item["expected_sources"]

            # ----------------------------------------
            # Retrieval timing
            # ----------------------------------------

            retrieval_start = time.perf_counter()

            if self.reranker:
                candidate_k = max(top_k * 3, 30)

                candidates = self.retriever.retrieve(
                    query=question,
                    top_k=candidate_k,
                )
            else:
                candidates = self.retriever.retrieve(
                    query=question,
                    top_k=top_k,
                )

            retrieval_time = time.perf_counter() - retrieval_start

            retrieval_times.append(retrieval_time)

            # ----------------------------------------
            # Reranking timing
            # ----------------------------------------

            rerank_time = 0.0

            if self.reranker:

                rerank_start = time.perf_counter()

                results = self.reranker.rerank(
                    query=question,
                    chunks=candidates,
                    top_k=top_k,
                    score_threshold=-100.0,
                )

                rerank_time = time.perf_counter() - rerank_start

            else:
                results = candidates

            rerank_times.append(rerank_time)

            # ----------------------------------------
            # Total pipeline retrieval time
            # ----------------------------------------

            total_time = retrieval_time + rerank_time

            total_times.append(total_time)

            # ----------------------------------------
            # Extract retrieved sources
            # ----------------------------------------

            retrieved_sources = []

            for result in results:

                source = result.metadata.get("source_file")

                if source and source not in retrieved_sources:
                    retrieved_sources.append(source)

            # ----------------------------------------
            # Metrics
            # ----------------------------------------

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

            # ----------------------------------------
            # Output
            # ----------------------------------------

            print(f"[{index}/{len(questions)}] " f"{question}")

            print(f"Recall@{top_k}: {recall:.2f}")

            print(f"Reciprocal Rank: {rr:.3f}")

            print(f"Retrieval Time: {retrieval_time:.3f}s")

            if self.reranker:
                print(f"Reranking Time: {rerank_time:.3f}s")

            print(f"Total Time: {total_time:.3f}s\n")

            print(f"Expected Sources: " f"{expected_sources}")

            print(f"Retrieved Sources: " f"{retrieved_sources}")

        # ----------------------------------------
        # Final evaluation results
        # ----------------------------------------

        results = {
            "total_questions": len(questions),
            f"recall_at_{top_k}": (calculate_average(recall_scores)),
            "mrr": calculate_average(reciprocal_ranks),
            "average_retrieval_time": (calculate_average(retrieval_times)),
            "average_rerank_time": (calculate_average(rerank_times)),
            "average_total_time": (calculate_average(total_times)),
        }

        return results
