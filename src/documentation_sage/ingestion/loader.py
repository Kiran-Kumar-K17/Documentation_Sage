from pathlib import Path
from langchain_community.document_loaders import TextLoader
from documentation_sage.core.ids import generate_document_id
from documentation_sage.schemas.documents import Document

class TextDocumentLoader:

    def load_directory(self, directory_path: Path) -> list[Document]:
        documents: list[Document] = []
        text_files = sorted(directory_path.glob("**/*.txt"))

        for text_file in text_files:
            loader = TextLoader(str(text_file))
            loaded_documents = loader.load()

            for loaded_document in loaded_documents:
                source_path = str(text_file)

                document = Document(
                            document_id=generate_document_id(source_path),
                            content=loaded_document.page_content,
                            metadata={
                                **loaded_document.metadata,
                                "source_file":text_file.name,
                                "source_path":source_path,
                                "file_type": "txt",
                            },
                            )
                documents.append(document)
        return documents
