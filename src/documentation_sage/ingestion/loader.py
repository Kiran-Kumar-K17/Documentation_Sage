from pathlib import Path

from langchain_community.document_loaders import TextLoader

from documentation_sage.core.ids import generate_document_id
from documentation_sage.schemas.documents import Document


class TextDocumentLoader:

    # Files that mainly contain navigation, indexes,
    # licenses, or metadata instead of useful documentation.
    EXCLUDED_FILES = {
        "contents.txt",
        "index.txt",
        "copyright.txt",
        "license.txt",
        "changelog.txt",
        "improve-page.txt",
        "improve-page-nojs.txt",
    }

    def load_directory(
        self,
        directory_path: Path,
    ) -> list[Document]:

        documents: list[Document] = []

        text_files = sorted(directory_path.glob("**/*.txt"))

        for text_file in text_files:

            # Skip noisy documentation files
            if text_file.name in self.EXCLUDED_FILES:
                print(f"Skipping: {text_file}")
                continue

            loader = TextLoader(
                str(text_file),
                encoding="utf-8",
            )

            loaded_documents = loader.load()

            for loaded_document in loaded_documents:

                source_path = str(text_file)

                document = Document(
                    document_id=generate_document_id(source_path),
                    content=loaded_document.page_content,
                    metadata={
                        **loaded_document.metadata,
                        "source_file": text_file.name,
                        "source_path": source_path,
                        "file_type": "txt",
                    },
                )

                documents.append(document)

        return documents
