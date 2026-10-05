import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate


CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "learnsmart_documents"
OUTPUT_FILE = Path("chat_output.txt")


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


def format_source(document):
    """Create a readable source citation with page number when available."""

    source = document.metadata.get("source", "Unknown source")

    page = document.metadata.get("page")

    if page is not None:
        return f"{Path(source).name} — Page {page + 1}"

    return Path(source).name


def ask_question(question, vector_store, llm):
    """Answer using RAG when relevant context is found,
    otherwise use the LLM's general knowledge.
    """

    results = vector_store.similarity_search_with_score(
        question,
        k=3,
    )

    relevant_results = [
        (document, score)
        for document, score in results
        if score < 1.2
    ]

    if relevant_results:
        context = "\n\n".join(
            document.page_content
            for document, _ in relevant_results
        )

        prompt = ChatPromptTemplate.from_template(
            """
You are a helpful RAG question-answering assistant.

Answer the user's question using ONLY the provided context.

Do not use outside knowledge when answering from the context.

If the context does not contain enough information, clearly say so.

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

        response = llm.invoke(messages)

        sources = []

        for document, _ in relevant_results:
            source = format_source(document)

            if source not in sources:
                sources.append(source)

        return response.content, sources, "RAG"

    general_prompt = ChatPromptTemplate.from_template(
        """
You are a helpful general-purpose AI assistant.

Answer the user's question clearly and accurately.

The question is not covered by the supplied document corpus,
so answer using your general language-model knowledge.

Question:
{question}

Answer:
"""
    )

    messages = general_prompt.format_messages(
        question=question,
    )

    response = llm.invoke(messages)

    return response.content, ["General LLM knowledge"], "GENERAL"


def main():
    print("=" * 60)
    print("          LEARNSMART HYBRID RAG CHATBOT")
    print("=" * 60)
    print("You can ask questions about LearnSmart or general questions.")
    print("Type 'exit' to stop the chatbot.\n")

    vector_store = get_vector_store()
    llm = get_llm()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        while True:
            question = input("\nAsk a question: ").strip()

            if question.lower() == "exit":
                print("\nChatbot stopped.")
                break

            if not question:
                print("Please enter a question.")
                continue

            answer, sources, answer_type = ask_question(
                question,
                vector_store,
                llm,
            )

            print("\nAnswer:")
            print(answer)

            print(f"\nAnswer Type: {answer_type}")

            print("\nSources:")
            for source in sources:
                print(f"- {source}")

            file.write("=" * 60 + "\n")
            file.write(f"Question: {question}\n\n")
            file.write(f"Answer: {answer}\n\n")
            file.write(f"Answer Type: {answer_type}\n\n")
            file.write("Sources:\n")

            for source in sources:
                file.write(f"- {source}\n")

            file.write("\n")

    print(f"Conversation saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()