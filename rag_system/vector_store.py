"""
ChromaDB vector store management.

Handles ingestion (add documents) and retrieval (similarity search).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_system.config import settings
from rag_system.embeddings import get_embeddings
from rag_system.logger import get_logger

logger = get_logger(__name__)

_vectorstore: Optional[Chroma] = None


def _ensure_persist_dir() -> Path:
    """Create the ChromaDB persist directory if it doesn't exist."""
    path = settings.chroma_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_vectorstore(force_new: bool = False) -> Chroma:
    """
    Return a Chroma vector store instance.
    """
    global _vectorstore

    if _vectorstore is None or force_new:
        persist_dir = _ensure_persist_dir()
        logger.info("Initialising ChromaDB at %s", persist_dir)

        client = chromadb.PersistentClient(path=str(persist_dir))
        _vectorstore = Chroma(
            client=client,
            collection_name=settings.chroma_collection_name,
            embedding_function=get_embeddings(),
        )
        logger.info("ChromaDB ready (collection: %s)", settings.chroma_collection_name)

    return _vectorstore


def add_documents(chunks: List[Document]) -> int:
    """
    Add document chunks to the vector store.

    Returns the number of chunks added.
    """
    if not chunks:
        logger.warning("No chunks to add")
        return 0

    store = get_vectorstore()
    store.add_documents(chunks)
    logger.info("Added %d chunks to the vector store", len(chunks))
    return len(chunks)


def similarity_search(query: str, k: Optional[int] = None) -> List[Document]:
    """
    Retrieve the *k* most relevant chunks for a query.
    Auto-reconnects if out-of-sync with disk.
    """
    global _vectorstore
    k = k or settings.retrieval_top_k
    try:
        store = get_vectorstore()
        results = store.similarity_search(query, k=k)
    except Exception as exc:
        logger.warning("Similarity search cache miss/stale (%s), reloading store...", exc)
        _vectorstore = None
        store = get_vectorstore(force_new=True)
        results = store.similarity_search(query, k=k)

    logger.info("Retrieved %d chunks for query (top_k=%d)", len(results), k)
    return results


def similarity_search_with_scores(
    query: str, k: Optional[int] = None
) -> List[tuple[Document, float]]:
    """
    Retrieve the *k* most relevant chunks with their distance scores.
    Auto-reconnects if out-of-sync with disk.
    """
    global _vectorstore
    k = k or settings.retrieval_top_k
    try:
        store = get_vectorstore()
        results = store.similarity_search_with_score(query, k=k)
    except Exception as exc:
        logger.warning("Scored search cache miss/stale (%s), reloading store...", exc)
        _vectorstore = None
        store = get_vectorstore(force_new=True)
        results = store.similarity_search_with_score(query, k=k)

    logger.info("Retrieved %d scored chunks for query", len(results))
    return results


def get_collection_stats() -> Dict:
    """
    Return basic stats about the current collection.
    """
    global _vectorstore
    try:
        store = get_vectorstore()
        count = store._collection.count()
    except Exception:
        _vectorstore = None
        store = get_vectorstore(force_new=True)
        count = store._collection.count()

    return {
        "collection_name": settings.chroma_collection_name,
        "document_count": count,
        "persist_directory": str(settings.chroma_path),
    }


def get_documents_summary() -> List[Dict]:
    """
    Return a list of unique indexed documents with chunk counts and types.
    """
    global _vectorstore
    try:
        store = get_vectorstore()
        data = store._collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
    except Exception:
        try:
            _vectorstore = None
            store = get_vectorstore(force_new=True)
            data = store._collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])
        except Exception:
            return []

    from collections import Counter
    counts = Counter()
    types = {}
    for m in metadatas:
        if not m:
            continue
        src = m.get("source", "unknown")
        clean_name = Path(src).name if src != "unknown" else "unknown"
        counts[clean_name] += 1
        if "type" in m:
            types[clean_name] = m["type"]
        else:
            types[clean_name] = Path(clean_name).suffix.lstrip(".").lower() or "txt"

    docs = []
    for name, chunk_cnt in counts.items():
        docs.append({
            "name": name,
            "chunks": chunk_cnt,
            "type": types.get(name, "doc")
        })
    return docs


def clear_collection() -> None:
    """
    Delete all documents from the collection.
    """
    global _vectorstore

    persist_dir = _ensure_persist_dir()
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        client.delete_collection(settings.chroma_collection_name)
        logger.info("Deleted collection '%s'", settings.chroma_collection_name)
    except ValueError:
        logger.info("Collection '%s' does not exist, nothing to clear", settings.chroma_collection_name)

    # Reset singleton so it gets re-created on next access
    _vectorstore = None
