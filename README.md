# Documentation Sage

Documentation Sage is a modular Retrieval-Augmented Generation (RAG) system for answering questions from Python technical documentation. It combines semantic vector search and BM25 keyword search, then improves final relevance with Reciprocal Rank Fusion (RRF) and cross-encoder reranking before generating an answer with a local Ollama model.

## Highlights

- Ingests documentation files from a directory
- Splits content with recursive chunking
- Creates Sentence Transformer embeddings
- Stores vectors persistently in ChromaDB
- Builds and persists a BM25 keyword index
- Combines vector and BM25 results with Reciprocal Rank Fusion
- Reranks candidates with a cross-encoder
- Generates answers locally through Ollama (`phi4-mini:3.8b`)
- Evaluates the final retrieval pipeline with Recall@K and MRR

## Architecture

```text
Python documentation
        |
        v
Document loader -> Recursive chunker
        |
        +-------------------------+
        |                         |
        v                         v
Sentence Transformer           BM25 index
        |                         |
        v                         v
ChromaDB vector store      Keyword retrieval
        |                         |
        +-----------+-------------+
                    v
         RRF hybrid retrieval
                    |
                    v
          Cross-encoder reranker
                    |
                    v
             Context builder
                    |
                    v
        Ollama local LLM -> Answer
```

## Retrieval flow

```text
Question
  -> vector search + BM25 search
  -> Reciprocal Rank Fusion
  -> candidate chunks
  -> cross-encoder reranking
  -> top relevant chunks
  -> context
  -> Ollama answer
```

## Tech stack

| Area              | Technology                   |
| ----------------- | ---------------------------- |
| Language          | Python                       |
| Embeddings        | Sentence Transformers        |
| Vector database   | ChromaDB                     |
| Keyword retrieval | BM25                         |
| Hybrid ranking    | Reciprocal Rank Fusion (RRF) |
| Reranking         | Cross-Encoder                |
| Local generation  | Ollama                       |
| Generation model  | `phi4-mini:3.8b`             |

## Project structure

```text
Documentation_Sage/
├── data/
│   ├── python/                # Source documentation
│   ├── indexes/               # Persisted BM25 index
│   └── vector_store/          # Persistent ChromaDB files
├── src/documentation_sage/
│   ├── chunking/
│   ├── core/
│   ├── embeddings/
│   ├── evaluation/
│   ├── generation/
│   ├── ingestion/
│   ├── pipelines/
│   ├── rerankers/
│   ├── retrievers/
│   └── vectorstores/
├── evaluation.json
├── pyproject.toml
└── README.md
```

## Setup

### Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone https://github.com/Kiran-Kumar-K17/Documentation_Sage.git
cd Documentation_Sage
uv venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate
```

Install the project dependencies:

```bash
uv add -r requirements.txt
```

Install Ollama, then download the model used by the project:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
ollama pull phi4-mini:3.8b
```

Start the Ollama service if it is not already running:

```bash
ollama serve
```

## Run

```bash
python src/documentation_sage/main.py
```

On the first run, the application loads the documentation, creates chunks, builds the BM25 index, creates embeddings, and persists them in ChromaDB. Later runs reuse the saved indexes.

Example question:

```text
What is a class in Python?
```

## Evaluation

Run the benchmark with:

```bash
python src/documentation_sage/evaluate.py
```

The evaluation measures the final retrieval pipeline—hybrid retrieval followed by cross-encoder reranking—using:

- **Recall@K**: the share of expected relevant sources found within the top _K_ results.
- **MRR (Mean Reciprocal Rank)**: how highly the first relevant result is ranked.

### Results

| Metric              |     Result |
| ------------------- | ---------: |
| Questions evaluated |         10 |
| Indexed chunks      |     15,886 |
| Recall@10           | **81.67%** |
| MRR                 | **0.5511** |

The hybrid RAG retrieval system was evaluated using 10 representative questions from the Python documentation dataset. The system achieved a Recall@10 of 81.67% and an MRR of 0.5511. Recall@10 improved from 75% to 81.67% during development. Retrieval, reranking, and total pipeline latency were measured separately to ensure that cross-encoder reranking time was not incorrectly reported as retrieval latency.

## How hybrid retrieval works

Vector search identifies semantically related chunks, while BM25 identifies chunks that share important query terms. RRF combines their rankings:

```text
RRF score = sum(1 / (k + rank))
```

The cross-encoder then scores each query-and-chunk pair together and returns the strongest final results.

## Status

Documentation Sage is complete as Version 1: a portfolio-ready implementation of an end-to-end, locally generated RAG workflow with hybrid retrieval, reranking, persistent indexing, and evaluation.

## License

This project is intended for educational and portfolio use. Add a license file before distributing it under a specific license.
