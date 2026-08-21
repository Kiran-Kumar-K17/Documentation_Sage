from documentation_sage.schemas.retrieval import RetrievedChunk


def build_context(
    chunks: list[RetrievedChunk],
    max_chunk_chars: int = 1500,
    max_context_chars: int = 6000,
) -> str:

    context_parts = []
    current_length = 0

    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")

        content = chunk.content[:max_chunk_chars]

        chunk_text = f"""[Source: {source}]

{content}
"""

        if current_length + len(chunk_text) > max_context_chars:
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)

    return "\n\n".join(context_parts)
