RAG System

A document-grounded question-answering service that retrieves relevant passages from your files, uses them to generate an answer, and can return the supporting sources.












































































Live service: 
rag-system-1rui.onrender.com · Interactive API documentation: 
Swagger UI

Overview

General-purpose language models do not automatically know the contents of your private or domain-specific documents. This project addresses that gap with a Retrieval-Augmented Generation (RAG) pipeline: it ingests documents, divides them into overlapping chunks, creates local embeddings, retrieves relevant passages for each question, and sends that context to an LLM. Responses can include source information so users can inspect the material behind an answer.

The same pipeline is available through a FastAPI REST API and a command-line interface. The service is deployed on Render, with a persistent disk for the vector store and embedding-model cache.

Highlights

•
Ingests PDF, TXT, DOCX, Markdown, CSV, and HTML files.

•
Uses Hugging Face all-MiniLM-L6-v2 embeddings locally; embedding generation does not require a separate API key.

•
Stores and searches document vectors with ChromaDB.

•
Generates responses using Groq-hosted LLM inference.

•
Returns supporting source chunks with query responses when requested.

•
Provides both a REST API with interactive Swagger documentation and a CLI built with Click and Rich.

•
Protects ingestion and reset operations with an X-Admin-Key header.

•
Configures model, chunking, retrieval, storage, and server settings through environment variables.

How it works

mermaid

Source



Ingestion: Files are parsed according to format and split into overlapping chunks. The default chunk size is 1,000 characters with 200 characters of overlap. Chunks are embedded locally and stored in ChromaDB.

Retrieval and generation: A question is embedded and compared with the indexed chunks. The default is to retrieve the five most similar chunks. LangChain assembles the retrieved context into the generation request, and the response can include the supporting sources.

Technology

Area
Technology
API
FastAPI, Uvicorn
RAG orchestration
LangChain
LLM inference
Groq
Embeddings
Hugging Face all-MiniLM-L6-v2 (local)
Vector store
ChromaDB
CLI
Click, Rich
Configuration
Pydantic Settings
Hosting
Render with persistent disk




Try the live service

The hosted API is available at https://rag-system-1rui.onrender.com.

Bash


# Check service health
curl https://rag-system-1rui.onrender.com/health

# Inspect vector-store statistics
curl https://rag-system-1rui.onrender.com/api/status

# Ask a question and request sources
curl -X POST https://rag-system-1rui.onrender.com/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this about?","return_sources":true}'



For interactive request and response schemas, open the 
Swagger UI.

API reference

Method
Endpoint
Authentication
Purpose
GET
/health
None
Health check
POST
/api/ingest
X-Admin-Key
Upload and ingest a file (10 MB maximum)
POST
/api/ingest/directory
X-Admin-Key
Ingest a directory available on the server
POST
/api/query
None
Ask a question about indexed documents
GET
/api/status
None
View vector-store statistics
DELETE
/api/reset
X-Admin-Key
Clear the indexed documents




Example document upload (requires an admin key):

Bash


curl -X POST https://rag-system-1rui.onrender.com/api/ingest \
  -H "X-Admin-Key: <your-admin-key>" \
  -F "file=@document.pdf"



Run locally

Prerequisites

•
Python 3.x

•
A 

Groq API key

Installation

Bash


git clone https://github.com/balaravi444/RAG_SYSTEM.git
cd RAG_SYSTEM
python -m venv .venv



Activate the virtual environment:

Bash


# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate



Install the project and configure the environment:

Bash


pip install -e .
cp .env.example .env



Edit .env and set GROQ_API_KEY and a long, randomly generated ADMIN_API_KEY. Do not commit real credentials.

CLI

Bash


rag ingest ./my_docs/                   # Ingest a file or directory recursively
rag ask "What are the conclusions?"     # Ask one question
rag ask "Summarize the findings" --sources -k 10
rag chat                                # Start an interactive session
rag status                              # Show vector-store statistics
rag reset                               # Clear the knowledge base



REST API

Bash


python -m rag_system



Alternatively, run the development server with automatic reload:

Bash


uvicorn rag_system.api:app --reload --port 8000



Local API documentation: 

http://localhost:8000/docs.

Configuration

Settings can be supplied as environment variables or in a local .env file.

Variable
Default
Description
GROQ_API_KEY
Required
API key for Groq inference
ADMIN_API_KEY
Required for admin actions
Protects ingestion and reset operations
LLM_MODEL_NAME
openai/gpt-oss-120b
Groq model identifier
EMBEDDING_MODEL_NAME
all-MiniLM-L6-v2
Hugging Face embedding model
CHROMA_PERSIST_DIR
./data/chroma_db
ChromaDB persistence directory
CHROMA_COLLECTION_NAME
rag_documents
ChromaDB collection name
CHUNK_SIZE
1000
Maximum characters per chunk
CHUNK_OVERLAP
200
Overlap between consecutive chunks
RETRIEVAL_TOP_K
5
Number of chunks retrieved per query
API_HOST
127.0.0.1
API bind host
API_PORT
8000
API bind port
LOG_LEVEL
INFO
Logging verbosity




Groq's available models can change. Check the 

current model list before changing LLM_MODEL_NAME.

Deployment on Render

The repository includes a render.yaml Blueprint to define the deployment configuration as code.

Setting
Configuration
Runtime
Python
Plan
Standard (the embedding model and dependencies require more than 512 MB RAM)
Build command
pip install .
Start command
uvicorn rag_system.api:app --host 0.0.0.0 --port $PORT
Health-check path
/health
Persistent disk
1 GB mounted at /var/data
Required secrets
GROQ_API_KEY, ADMIN_API_KEY (set in Render; do not commit)




The persistent disk stores ChromaDB data at /var/data/chroma_db and the Hugging Face model cache at /var/data/huggingface. This allows indexed documents to survive service restarts and avoids downloading the embedding model on every restart.

To deploy your own instance, create a new Blueprint in the Render dashboard, select this repository, and configure the two required secrets.

Project structure

Plain Text


RAG_SYSTEM/
├── rag_system/
│   ├── __main__.py          # python -m rag_system entry point
│   ├── api.py               # FastAPI REST API
│   ├── cli.py               # Click CLI and Rich output
│   ├── config.py            # Pydantic settings and environment configuration
│   ├── document_loader.py   # Document loading and chunking
│   ├── embeddings.py        # Hugging Face embeddings
│   ├── logger.py            # Logging setup
│   ├── pipeline.py          # Shared pipeline orchestration
│   ├── rag_chain.py         # LangChain RAG chain and Groq integration
│   └── vector_store.py      # ChromaDB operations
├── .env.example             # Environment-variable template
├── render.yaml              # Render Blueprint configuration
├── pyproject.toml           # Dependencies and project metadata
└── README.md



The project separates document loading, embedding, storage, generation, and user interfaces into focused modules. The API and CLI both use the shared pipeline in pipeline.py.

Security and current limitations

•
Admin operations are protected: ingestion, directory ingestion, and reset require X-Admin-Key. Individual uploads are limited to 10 MB.

•
Read endpoints are public: querying, status, health, and Swagger documentation do not require user authentication. The admin key does not make indexed documents private. Do not use this deployment for sensitive documents.

•
Secrets are configured outside source control: provide credentials through environment variables and keep .env out of version control.

•
The vector store is single-node: ChromaDB runs within one service instance on a persistent disk. This keeps deployment straightforward but does not provide horizontal scaling.

Planned improvements

Potential next steps include user authentication and per-user collections, public-endpoint rate limiting, streaming responses, hybrid retrieval and reranking, a retrieval-quality evaluation harness, automated tests and CI, and Docker support.

License

Released under the MIT License.

Author

Bala Ravi · 

GitHub

