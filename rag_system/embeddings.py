"""
Embedding model wrapper.

Uses HuggingFace sentence-transformers for local, free embeddings.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from rag_system.config import settings
from rag_system.logger import get_logger

logger = get_logger(__name__)

_embeddings_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a singleton embedding model.

    The model is downloaded on first call and cached locally by
    sentence-transformers.
    """
    global _embeddings_instance

    if _embeddings_instance is None:
        logger.info("Loading embedding model: %s", settings.embedding_model_name)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
        logger.info("Embedding model loaded successfully")

    return _embeddings_instance
