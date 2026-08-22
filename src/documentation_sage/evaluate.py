from pathlib import Path

from documentation_sage.main import (
    create_rag_pipeline,
)
from documentation_sage.evaluation.evaluator import (
    RAGEvaluator,
)


def main():

    print("Initializing Documentation Sage...")

    rag = create_rag_pipeline()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    evaluator = RAGEvaluator(
        retriever=rag.retriever,
        reranker=rag.reranker,
    )

    questions_path = PROJECT_ROOT / "data" / "evaluation" / "evaluation.json"

    questions = evaluator.load_questions(questions_path)

    results = evaluator.evaluate(
        questions=questions,
        top_k=10,
    )

    print("\n" + "=" * 60)

    print("RAG EVALUATION RESULTS")

    print("=" * 60)

    print(f"Questions Tested: " f"{results['total_questions']}")

    print(f"Recall@10: " f"{results['recall_at_10'] * 100:.2f}%")

    print(f"MRR: " f"{results['mrr']:.4f}")

    print(f"Average Retrieval Time: " f"{results['average_retrieval_time']:.3f}s")

    print("=" * 60)


if __name__ == "__main__":
    main()
