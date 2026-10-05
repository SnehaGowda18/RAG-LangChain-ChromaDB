from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.chatbot import get_vector_store, get_llm, ask_question


app = FastAPI(
    title="LearnSmart Hybrid RAG API",
    description="RAG chatbot using LangChain, ChromaDB and Groq",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    answer_type: str
    sources: list[str]


# Load the vector store and LLM once when the API starts.
vector_store = get_vector_store()
llm = get_llm()


@app.get("/")
def root():
    return {
        "message": "LearnSmart Hybrid RAG API is running",
        "status": "success",
    }


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    answer, sources, answer_type = ask_question(
        question,
        vector_store,
        llm,
    )

    return AnswerResponse(
        question=question,
        answer=answer,
        answer_type=answer_type,
        sources=sources,
    )