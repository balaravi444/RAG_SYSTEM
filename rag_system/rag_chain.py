"""
RAG chain — ties together retrieval and generation via Groq.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import SecretStr

from rag_system.config import settings
from rag_system.logger import get_logger
from rag_system.vector_store import similarity_search_with_scores

logger = get_logger(__name__)

# ────────────────────────────────────────────────────────────
# Prompt template
# ────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are a helpful, accurate assistant. Answer the user's question using ONLY \
the context provided below. If the context doesn't contain enough information \
to fully answer, say so clearly — do not make up facts.

When possible, cite which source document your answer comes from.

CONTEXT:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


# ────────────────────────────────────────────────────────────
# LLM factory
# ────────────────────────────────────────────────────────────

_llm_instance: Optional[ChatGroq] = None


def get_llm() -> ChatGroq:
    """Return a singleton Groq chat model."""
    global _llm_instance

    if _llm_instance is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )

        _llm_instance = ChatGroq(
            api_key=SecretStr(settings.groq_api_key),
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        logger.info("Groq LLM ready (model: %s)", settings.llm_model_name)

    return _llm_instance


# ────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────

def _format_context(docs: List[Document]) -> str:
    """Combine retrieved documents into a single context string."""
    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        source = str(doc.metadata.get("source", "unknown")).replace("\\", "/")
        source = Path(source).name
        page = doc.metadata.get("page", "")
        ref = f"[Source: {source}"
        if page:
            ref += f", page {page}"
        ref += "]"
        parts.append(f"--- Chunk {i} {ref} ---\n{doc.page_content}")
    return "\n\n".join(parts)


# ────────────────────────────────────────────────────────────
# Core query function
# ────────────────────────────────────────────────────────────

def query(
    question: str,
    top_k: Optional[int] = None,
    return_sources: bool = False,
) -> Dict:
    """
    Run a full RAG query:
      1. Retrieve relevant chunks from the vector store
      2. Build a context-enriched prompt
      3. Send to Groq for generation

    Returns a dict with the answer and optionally the source documents.
    """
    logger.info("RAG query: %s", question[:120])

    # 1. Retrieve
    scored_docs = similarity_search_with_scores(question, k=top_k)
    docs = [doc for doc, _ in scored_docs]

    if not docs:
        return {
            "answer": "I couldn't find any relevant information in the knowledge base. "
                      "Please make sure documents have been ingested first.",
            "sources": [],
        }

    # 2. Format context
    context = _format_context(docs)

    # 3. Generate
    llm = get_llm()
    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    result: Dict = {"answer": answer}

    if return_sources:
        result["sources"] = [
            {
                "content": doc.page_content[:300],
                "metadata": doc.metadata,
                "score": round(float(score), 4),
            }
            for doc, score in scored_docs
        ]

    logger.info("Generated answer (%d chars)", len(answer))
    return result


def query_with_chain(question: str, top_k: Optional[int] = None) -> str:
    """
    Simplified query that returns just the answer string.
    Useful for the LCEL chain-style interface.
    """
    result = query(question, top_k=top_k, return_sources=False)
    return result["answer"]
