from pathlib import Path

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"


def load_documents():
    """Load TXT and PDF documents from the data folders."""

    documents = []

    # Load TXT files from data/
    for file_path in DATA_DIR.glob("*.txt"):
        loader = TextLoader(
            str(file_path),
            encoding="utf-8",
        )

        documents.extend(loader.load())

    # Load PDF files from data/pdfs/
    for file_path in PDF_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        documents.extend(loader.load())

    return documents


def load_and_split_documents():
    """Load documents and split them into smaller chunks."""

    documents = load_documents()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":
    documents = load_documents()
    chunks = load_and_split_documents()

    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    for index, chunk in enumerate(chunks[:5], start=1):
        print(f"\n--- Chunk {index} ---")
        print(chunk.page_content[:500])
        print(f"Source: {chunk.metadata.get('source')}")