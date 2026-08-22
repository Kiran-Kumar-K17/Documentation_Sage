from langchain_text_splitters import RecursiveCharacterTextSplitter
from documentation_sage.schemas.documents import Document, Chunk
from langchain_core.documents import Document as LangChainDocument
from documentation_sage.core.ids import generate_chunk_id
from documentation_sage.core.config import ChunkingConfig

class RecursiveChunker:

    def __init__(self, config: ChunkingConfig):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
    def split(self, documents: list[Document]) -> list[Chunk]:

        chunks: list[Chunk] = []

        for document in documents:
            langchain_document = LangChainDocument(
                page_content=document.content,
                metadata=document.metadata,
            )

            split_documents = self.text_splitter.split_documents([langchain_document])

            for chunk_index, split_document in enumerate(split_documents):

                raw_content = split_document.page_content
                source_file = split_document.metadata.get("source_file", "")
                if source_file:
                    content = f"Document: {source_file}\n\n{raw_content}"
                else:
                    content = raw_content

                chunk_id = generate_chunk_id(
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    content=content
                )
                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    content=content,
                    metadata=split_document.metadata,
                )
                chunks.append(chunk)
            
        return chunks