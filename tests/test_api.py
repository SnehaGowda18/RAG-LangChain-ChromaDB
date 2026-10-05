from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"


def test_rag_question():
    response = client.post(
        "/ask",
        json={
            "question": "What should a beginner learn in Python?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What should a beginner learn in Python?"
    assert data["answer"]
    assert data["answer_type"] in ["RAG", "GENERAL"]
    assert len(data["sources"]) > 0


def test_general_question():
    response = client.post(
        "/ask",
        json={
            "question": "What is the capital of Japan?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"]
    assert data["answer_type"] in ["RAG", "GENERAL"]
    assert len(data["sources"]) > 0


def test_empty_question():
    response = client.post(
        "/ask",
        json={
            "question": ""
        },
    )

    assert response.status_code == 400


def test_pdf_question():
    response = client.post(
        "/ask",
        json={
            "question": "What are the important Python topics beginners should learn?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"]
    assert data["answer_type"] == "RAG"

    assert any(
        "LearnSmart Course Guide.pdf" in source
        for source in data["sources"]
    )