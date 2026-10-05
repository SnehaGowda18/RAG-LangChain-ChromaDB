from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from document_loader import load_and_split_documents


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "learnsmart_documents"


def create_vector_store():
    """Create a ChromaDB vector store from document chunks."""

    chunks = load_and_split_documents()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    return vector_store


if __name__ == "__main__":
    vector_store = create_vector_store()

    print("ChromaDB vector store created successfully!")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database directory: {CHROMA_DIR}")