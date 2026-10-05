from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "learnsmart_documents"


def get_vector_store():
    """Load the existing ChromaDB vector store."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    return vector_store


def retrieve_documents(query, k=3):
    """Retrieve the most relevant document chunks."""

    vector_store = get_vector_store()

    results = vector_store.similarity_search(
        query,
        k=k,
    )

    return results


if __name__ == "__main__":
    query = "What should a beginner learn in Python?"

    results = retrieve_documents(query)

    print(f"Query: {query}")
    print(f"Retrieved documents: {len(results)}")

    for index, document in enumerate(results, start=1):
        print(f"\n--- Result {index} ---")
        print(document.page_content)
        print(f"Source: {document.metadata.get('source')}")