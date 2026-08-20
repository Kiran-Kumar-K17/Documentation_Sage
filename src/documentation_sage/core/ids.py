import hashlib

def generate_document_id(source_path: str) -> str:
    """
    Generate a deterministic ID for a source document.

    The same source path will always produce the same ID.
    """

    content = source_path.encode("utf-8")

    return hashlib.sha256(content).hexdigest()

def generate_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
     """
    Generate a deterministic ID for a document chunk.

    The same document, chunk index, and content
    will always produce the same ID.
    """
     content = f"{document_id}:{chunk_index}:{content}"

     return hashlib.sha256(content.encode("utf-8")).hexdigest()