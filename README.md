# RAG Pipeline with LangChain, ChromaDB & Groq

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) question-answering system using **LangChain, ChromaDB, HuggingFace embeddings, and Groq LLM**.

The system loads information from TXT and PDF documents, splits the documents into smaller chunks, converts the chunks into vector embeddings, stores them in ChromaDB, retrieves the most relevant information for a user question, and generates an answer using the Groq LLM.

The project also provides a **FastAPI REST API** and includes automated tests.

---

## Demo Video

A complete demonstration of the project is available here:

Watch the Demo Video : https://drive.google.com/file/d/1GiC7KkIx6U6ca97HZjbeZV9-_LJ3bsGd/view?usp=sharing
---

## Key Features

* TXT and PDF document loading
* Document chunking using LangChain
* HuggingFace sentence embeddings
* Persistent ChromaDB vector database
* Semantic similarity search
* Retrieval-Augmented Generation
* Groq LLM integration
* Source citations for retrieved answers
* General LLM fallback for questions outside the document corpus
* Interactive command-line chatbot
* FastAPI REST API
* Swagger API documentation
* Automated API and RAG tests
* Conversation output saved to a text file

---

## Technologies Used

| Technology            | Purpose                              |
| --------------------- | ------------------------------------ |
| Python                | Main programming language            |
| LangChain             | RAG pipeline and document processing |
| ChromaDB              | Vector database                      |
| HuggingFace           | Text embeddings                      |
| Sentence Transformers | `all-MiniLM-L6-v2` embedding model   |
| Groq                  | Large Language Model                 |
| FastAPI               | REST API                             |
| Uvicorn               | API server                           |
| PyPDF                 | PDF document loading                 |
| Pytest                | Automated testing                    |
| Git & GitHub          | Version control                      |

---

## Project Architecture

The main RAG workflow is:

```text
Documents
    ↓
Document Loading
    ↓
Text Chunking
    ↓
HuggingFace Embeddings
    ↓
ChromaDB Vector Store
    ↓
Similarity Retrieval
    ↓
Relevant Context
    ↓
Groq LLM
    ↓
Generated Answer
    ↓
Source Citations
```

---

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
│   ├── rag_pipeline.py
│   ├── retriever.py
│   └── vector_store.py
│
├── tests/
│   └── test_api.py
│
├── chroma_db/
│
├── .env
├── .gitignore
├── requirements.txt
├── pytest.ini
├── output.txt
├── chat_output.txt
└── README.md
```

> `.env`, `.venv`, and `chroma_db` are excluded from GitHub using `.gitignore` where appropriate.

---

## Dataset / Documents

The project uses a small document corpus containing:

* LearnSmart course information
* Python learning guidance
* Course-related information
* TXT document content
* PDF document content

The PDF is loaded page by page and both document types are converted into LangChain documents.

---

## Document Processing

The document loader reads TXT and PDF files from the `data` directory.

Documents are then split using:

```text
Chunk size: 500
Chunk overlap: 100
```

During testing, the project loaded:

```text
Documents loaded: 11
Total chunks created: 32
```

The chunks contain metadata such as their original source file and PDF page number.

---

## Embeddings

The project uses the HuggingFace embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The embedding model converts text chunks into numerical vector representations.

These vectors allow the system to perform semantic similarity search.

---

## ChromaDB Vector Store

The generated embeddings are stored in a persistent ChromaDB collection.

Collection name:

```text
learnsmart_documents
```

Database directory:

```text
chroma_db
```

The vector database is then used by the retriever to find the most relevant document chunks for a user question.

---

## Retrieval

The retriever performs similarity search against ChromaDB.

The system retrieves the top relevant documents for each question.

Example:

```text
Query:
What should a beginner learn in Python?

Retrieved documents: 3
```

The retrieved documents are then passed to the RAG pipeline.

---

## RAG Chatbot

The chatbot uses the retrieved document context to generate answers.

For questions covered by the document corpus, the system uses:

```text
Answer Type: RAG
```

The response also includes source citations such as:

```text
LearnSmart Course Guide.pdf — Page 1
learnsmart_guide.txt
```

If the document corpus does not contain relevant information, the system uses the Groq LLM as a general-purpose fallback.

Example:

```text
Question:
What is the capital of Japan?

Answer Type:
GENERAL

Source:
General LLM knowledge
```

This creates a hybrid RAG and general-purpose chatbot.

---

## Groq LLM

The project uses the Groq API through LangChain.

The configured model is:

```text
openai/gpt-oss-20b
```

The API key is stored securely in the `.env` file.

Example:

```env
GROQ_API_KEY=your_api_key_here
```

The API key is **not included in the GitHub repository**.

---

## FastAPI REST API

The project exposes the RAG chatbot through a FastAPI application.

Start the API with:

```powershell
uvicorn src.api:app --reload
```

The API runs locally at:

```text
http://127.0.0.1:8000
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

#### GET `/`

Checks whether the API is running.

Example response:

```json
{
  "message": "LearnSmart Hybrid RAG API is running",
  "status": "success"
}
```

#### POST `/ask`

Accepts a user question and returns the generated answer, answer type, and sources.

Example request:

```json
{
  "question": "What should a beginner learn in Python?"
}
```

Example response structure:

```json
{
  "question": "What should a beginner learn in Python?",
  "answer": "Generated answer...",
  "answer_type": "RAG",
  "sources": [
    "LearnSmart Course Guide.pdf — Page 1"
  ]
}
```

---

## Installation

### 1. Clone the Repository

```powershell
git clone https://github.com/SnehaGowda18/RAG-LangChain-ChromaDB.git
```

Move into the project directory:

```powershell
cd RAG-LangChain-ChromaDB
```

### 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

### 3. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Configure the Groq API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

Do not commit the `.env` file to GitHub.

---

## Running the Project

### Step 1: Load Documents

```powershell
python src\document_loader.py
```

This loads the TXT and PDF documents and creates text chunks.

### Step 2: Create ChromaDB Vector Store

```powershell
python src\vector_store.py
```

Expected output:

```text
ChromaDB vector store created successfully!
Collection: learnsmart_documents
```

### Step 3: Test Retrieval

```powershell
python src\retriever.py
```

This retrieves the most relevant document chunks for a sample question.

### Step 4: Run the Chatbot

```powershell
python src\chatbot.py
```

The chatbot accepts questions interactively.

Type:

```text
exit
```

to stop the chatbot.

Conversation results are also saved to:

```text
chat_output.txt
```

### Step 5: Run the FastAPI Application

```powershell
uvicorn src.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---

## Testing

The project includes automated tests using Pytest.

Run:

```powershell
python -m pytest -q
```

The final test suite contains five tests covering:

* API root endpoint
* RAG question answering
* General LLM fallback
* Empty question validation
* PDF-based RAG retrieval and source citation

Expected result:

```text
5 passed
```

---

## Source Citations

The chatbot tracks the source documents used to generate RAG responses.

For PDF documents, the response identifies the source page.

Example:

```text
LearnSmart Course Guide.pdf — Page 1
```

This improves transparency by allowing users to identify where the retrieved information came from.

---

## Security

The Groq API key is stored in an environment file:

```text
.env
```

The `.env` file is excluded from Git tracking using:

```text
.env
```

The following project-generated or environment-specific files are also excluded where appropriate:

```text
.venv/
__pycache__/
*.pyc
chroma_db/
.pytest_cache/
```

No API key or secret credentials are included in the public repository.

---

## Output Files

The project generates useful output files for demonstration and testing.

### `output.txt`

Contains project execution/output evidence.

### `chat_output.txt`

Contains chatbot questions, answers, answer types, and source information.

---

## Challenges

Some of the main implementation challenges included:

1. Loading and processing both TXT and PDF documents.
2. Splitting documents into useful chunks while maintaining context.
3. Creating and persisting embeddings in ChromaDB.
4. Connecting ChromaDB retrieval with the Groq LLM.
5. Implementing source citations for retrieved information.
6. Handling questions outside the document corpus using a general LLM fallback.
7. Exposing the RAG pipeline through a FastAPI REST API.
8. Testing both RAG and general-purpose responses.

---

## Future Improvements

Possible future improvements include:

* Support for additional document formats
* Conversation memory
* Improved retrieval and reranking
* Streaming LLM responses
* Authentication for the API
* Deployment to a cloud platform
* Web-based chatbot interface
* More advanced evaluation metrics
* Larger document collections
* Improved citation and relevance scoring

---

## Conclusion

This project demonstrates a complete Retrieval-Augmented Generation pipeline using modern open-source AI tools.

It combines document processing, embeddings, ChromaDB vector search, LangChain, Groq LLM generation, source citations, general-question fallback, FastAPI integration, and automated testing into a single working application.

The project provides both an interactive chatbot and a REST API for querying the document knowledge base.

---

## GitHub Repository

View the GitHub Repository : https://github.com/SnehaGowda18/RAG-LangChain-ChromaDB 

## Demo

Watch the Complete Demo Video : https://drive.google.com/file/d/1GiC7KkIx6U6ca97HZjbeZV9-_LJ3bsGd/view?usp=sharings
