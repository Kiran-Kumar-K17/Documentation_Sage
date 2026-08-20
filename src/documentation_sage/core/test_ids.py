from documentation_sage.core.ids import (
    generate_document_id,
    generate_chunk_id,
)


document_id_1 = generate_document_id(
    "../data/python/about.txt"
)

document_id_2 = generate_document_id(
    "../data/python/about.txt"
)

print("Document ID 1:", document_id_1)
print("Document ID 2:", document_id_2)

print(
    "Same document ID:",
    document_id_1 == document_id_2
)


chunk_id_1 = generate_chunk_id(
    document_id=document_id_1,
    chunk_index=0,
    content="Python provides the json module.",
)

chunk_id_2 = generate_chunk_id(
    document_id=document_id_1,
    chunk_index=0,
    content="Python provides the json module.",
)

print("\nChunk ID 1:", chunk_id_1)
print("Chunk ID 2:", chunk_id_2)

print(
    "Same chunk ID:",
    chunk_id_1 == chunk_id_2
)