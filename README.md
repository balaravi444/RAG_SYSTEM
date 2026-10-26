# 🧠 RAG System

A production-ready **Retrieval-Augmented Generation** system built with:

- **[Groq](https://groq.com)** — Ultra-fast LLM inference (Llama 3.1, Mixtral, etc.)
- **[ChromaDB](https://www.trychroma.com)** — Local vector database
- **[LangChain](https://python.langchain.com)** — LLM orchestration framework
- **[FastAPI](https://fastapi.tiangolo.com)** — REST API server
- **[HuggingFace](https://huggingface.co)** — Local embedding model (no API key needed)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 Multi-format ingestion | PDF, TXT, DOCX, Markdown, CSV, HTML |
| 🔍 Semantic search | Similarity-based retrieval using sentence-transformers |
| ⚡ Fast generation | Groq-powered LLM inference with sub-second latency |
| 🖥️ CLI interface | Beautiful terminal UI with Rich |
| 🌐 REST API | FastAPI with auto-generated Swagger docs |
| 💬 Interactive chat | Conversational Q&A mode |
| 📊 Source attribution | See which documents informed the answer |
| 🔧 Fully configurable | All settings via `.env` or environment variables |

## 📁 Project Structure

```
RAG_SYSTEM/
├── rag_system/
│   ├── __init__.py          # Package init
│   ├── __main__.py          # python -m rag_system entry point
│   ├── api.py               # FastAPI REST server
│   ├── cli.py               # Click CLI with Rich output
│   ├── config.py            # Pydantic settings (from .env)
│   ├── document_loader.py   # Multi-format doc loading & chunking
│   ├── embeddings.py        # HuggingFace embedding model
│   ├── logger.py            # Structured logging
│   ├── pipeline.py          # High-level orchestration
│   ├── rag_chain.py         # LangChain RAG chain with Groq
│   └── vector_store.py      # ChromaDB operations
├── .env.example             # Configuration template
├── .gitignore
├── pyproject.toml           # Project config & dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install the package and all dependencies
pip install -e .
```

### 2. Configure

```bash
# Copy the example config
cp .env.example .env

# Edit .env and add your Groq API key and a long, random admin key
# Get one free at: https://console.groq.com/keys
```

### 3. Ingest Documents

```bash
# Ingest a single file
rag ingest path/to/document.pdf

# Ingest an entire directory (recursive)
rag ingest path/to/documents/
```

### 4. Ask Questions

```bash
# Single question
rag ask "What is the main topic of the documents?"

# With source citations
rag ask "Summarize the key findings" --sources

# Interactive chat mode
rag chat
```

## 🖥️ CLI Reference

```
Usage: rag [OPTIONS] COMMAND [ARGS]...

Commands:
  ingest  📥 Ingest documents from a file or directory
  ask     ❓ Ask a question against the knowledge base
  chat    💬 Start an interactive chat session
  status  📊 Show vector store statistics
  reset   🗑️  Clear all documents from the vector store
```

### Examples

```bash
# Ingest documents
rag ingest ./my_docs/
rag ingest report.pdf

# Ask with custom retrieval count
rag ask "What are the conclusions?" -k 10

# Show sources with the answer
rag ask "Who authored section 3?" --sources

# Check what's in the store
rag status

# Clear everything
rag reset
```

## 🌐 REST API

### Start the server

```bash
# Option 1: Using the module
python -m rag_system

# Option 2: Using uvicorn directly
uvicorn rag_system.api:app --reload --port 8000
```

Then visit **http://localhost:8000/docs** for interactive Swagger documentation.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/ingest` | Upload & ingest a file (admin key required; 10 MB maximum) |
| `POST` | `/api/ingest/directory` | Ingest from server directory (admin key required) |
| `POST` | `/api/query` | Ask a question |
| `GET` | `/api/status` | Vector store stats |
| `DELETE` | `/api/reset` | Clear all documents (admin key required) |

### Example API Calls

```bash
# Upload a file
curl -X POST http://localhost:8000/api/ingest \
  -H "X-Admin-Key: your-admin-key" \
  -F "file=@document.pdf"

# Ask a question
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "return_sources": true}'

# Check status
curl http://localhost:8000/api/status
```

## ⚙️ Configuration

All settings can be configured via environment variables or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Your Groq API key |
| `LLM_MODEL_NAME` | `openai/gpt-oss-120b` | Groq model to use |
| `ADMIN_API_KEY` | (required for admin actions) | Protects uploads, directory ingestion, and reset |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | Where to store vectors |
| `CHROMA_COLLECTION_NAME` | `rag_documents` | ChromaDB collection name |
| `CHUNK_SIZE` | `1000` | Max characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `5` | Number of chunks to retrieve |
| `API_HOST` | `127.0.0.1` | FastAPI host |
| `API_PORT` | `8000` | FastAPI port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

Groq model availability changes over time; see the [current Groq model list](https://console.groq.com/docs/models) before changing `LLM_MODEL_NAME`.

## ☁️ Deploy on Render

This repository includes a `render.yaml` blueprint for a web service with a persistent ChromaDB disk. Connect the GitHub repository in Render and create a Blueprint Instance. Before the first deploy, set the `GROQ_API_KEY` and `ADMIN_API_KEY` secrets in the Render dashboard. Never commit `.env` or secret values.

The Render service and persistent disk use a paid plan. The persistent disk keeps indexed documents across deploys and restarts. If you choose a free, ephemeral deployment instead, documents can be lost when the service restarts or redeploys.

## 📝 License

MIT
