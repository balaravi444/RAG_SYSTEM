"""
High-level pipeline that orchestrates ingest and query workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from rag_system.document_loader import (
    chunk_documents,
    load_document,
    load_documents_from_directory,
)
from rag_system.logger import get_logger
from rag_system.rag_chain import query
from rag_system.vector_store import add_documents, clear_collection, get_collection_stats

logger = get_logger(__name__)


def ingest_file(file_path: str | Path) -> Dict:
    """
    Ingest a single file into the vector store.

    Returns stats about the ingestion.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    docs = load_document(path)
    chunks = chunk_documents(docs)
    added = add_documents(chunks)

    return {
        "file": str(path),
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunks_added": added,
    }


def ingest_directory(dir_path: str | Path) -> Dict:
    """
    Recursively ingest all supported files from a directory.

    Returns stats about the ingestion.
    """
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    docs = load_documents_from_directory(path)
    chunks = chunk_documents(docs)
    added = add_documents(chunks)

    return {
        "directory": str(path),
        "documents_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunks_added": added,
    }


def ask(
    question: str,
    top_k: Optional[int] = None,
    return_sources: bool = False,
) -> Dict:
    """
    Ask a question against the ingested knowledge base.

    Returns the LLM answer and optionally the source documents.
    """
    return query(question, top_k=top_k, return_sources=return_sources)


def status() -> Dict:
    """Return current vector store stats."""
    return get_collection_stats()


def list_documents() -> List[Dict]:
    """Return summary of all indexed documents in the vector store."""
    from rag_system.vector_store import get_documents_summary
    return get_documents_summary()


def reset() -> Dict:
    """Clear all documents from the vector store."""
    clear_collection()
    return {"status": "cleared", "message": "All documents have been removed from the vector store."}
