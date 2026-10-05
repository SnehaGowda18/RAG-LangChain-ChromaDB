import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "learnsmart_documents"
OUTPUT_FILE = Path("output.txt")


def get_vector_store():
    """Load the existing ChromaDB vector store."""

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def get_llm():
    """Create the Groq language model."""

    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the .env file.")

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key,
    )


def ask_question(question, k=3):
    """Retrieve relevant documents and generate a grounded answer."""

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        question,
        k=k,
    )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I could not find this information in the provided documents."

Do not use outside knowledge.
Do not invent information.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    messages = prompt.format_messages(
        context=context,
        question=question,
    )

    llm = get_llm()

    response = llm.invoke(messages)

    sources = []

    for document in documents:
        source = document.metadata.get("source", "Unknown source")

        if source not in sources:
            sources.append(source)

    return response.content, sources


def main():
    questions = [
        "What should a beginner learn in Python?",
        "What are the important topics in Data Structures and Algorithms?",
        "What are the common approaches in Machine Learning?",
        "What are the important areas of Cybersecurity?",
        "What is SQL used for?",
    ]

    output_lines = []

    output_lines.append("=" * 60)
    output_lines.append("RAG PIPELINE - QUESTION ANSWER RESULTS")
    output_lines.append("=" * 60)

    for index, question in enumerate(questions, start=1):

        print(f"\nProcessing Question {index}...")

        answer, sources = ask_question(question)

        output_lines.append("")
        output_lines.append("=" * 60)
        output_lines.append(f"QUESTION {index}")
        output_lines.append("=" * 60)
        output_lines.append(question)

        output_lines.append("")
        output_lines.append("ANSWER")
        output_lines.append("-" * 60)
        output_lines.append(answer)

        output_lines.append("")
        output_lines.append("SOURCES")
        output_lines.append("-" * 60)

        for source in sources:
            output_lines.append(f"- {source}")

    OUTPUT_FILE.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("All 5 questions processed successfully!")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()