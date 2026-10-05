# LearnSmart Hybrid RAG Pipeline

## Overview

LearnSmart Hybrid RAG is an AI-powered question-answering system built using Retrieval-Augmented Generation (RAG).

The system loads information from TXT and PDF documents, splits the content into smaller chunks, creates embeddings, stores them in ChromaDB, retrieves relevant information for a user question, and uses a Groq language model to generate an answer.

The chatbot also supports a general LLM fallback for questions that are not covered by the document corpus.

## Features

- TXT document loading
- PDF document loading
- Document chunking
- Hugging Face sentence-transformer embeddings
- ChromaDB vector storage
- Similarity-based document retrieval
- Groq LLM integration
- RAG-based question answering
- General LLM fallback
- Source citations
- PDF page-level source citations
- Interactive chatbot
- FastAPI REST API
- Swagger API documentation
- Automated testing

## Technologies Used

- Python
- LangChain
- ChromaDB
- Groq
- Hugging Face Sentence Transformers
- FastAPI
- Uvicorn
- PyPDF
- Pytest

## Project Structure

```text
RAG-LangChain-ChromaDB/
│
├── data/
│   ├── learnsmart_guide.txt
│   └── pdfs/
│       └── LearnSmart Course Guide.pdf
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── chatbot.py
│   ├── document_loader.py
│   ├── groq_check.py
│   ├── retriever.py
│   └── vector_store.py
│
├── tests/
│   └── test_api.py
│
├── chroma_db/
├── .env
├── .gitignore
├── requirements.txt
├── pytest.ini
├── output.txt
├── chat_output.txt
└── README.md