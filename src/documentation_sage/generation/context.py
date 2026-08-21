def build_context(
    chunks,
    max_context_chars: int = 8000,
    max_chunk_chars: int = 2000,
):
    context_parts = []
    used_chunks = []
    current_length = 0

    for chunk in chunks:
        content = chunk.content[:max_chunk_chars]

        if current_length + len(content) > max_context_chars:
            break

        source = chunk.metadata.get(
            "source_file",
            "Unknown source",
        )

        context_parts.append(f"[Source: {source}]\n\n{content}")

        used_chunks.append(chunk)

        current_length += len(content)

    context = "\n\n".join(context_parts)

    return context, used_chunks
