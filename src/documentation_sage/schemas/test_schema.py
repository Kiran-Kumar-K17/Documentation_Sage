from documentation_sage.schemas.documents import Document, Chunk
from documentation_sage.schemas.retrieval import RetrievedChunk


document = Document(
    document_id="python_json",
    content="Python provides the json module.",
    metadata={
        "source": "json.rst"
    }
)


chunk = Chunk(
    chunk_id="python_json_chunk_0",
    document_id=document.document_id,
    content="Python provides the json module.",
    chunk_index=0,
    metadata={
        "source": "json.rst"
    }
)


retrieved_chunk = RetrievedChunk(
    chunk_id=chunk.chunk_id,
    document_id=chunk.document_id,
    content=chunk.content,
    metadata=chunk.metadata,
    score=0.92,
    rank=1
)


print(document)
print()
print(chunk)
print()
print(retrieved_chunk)