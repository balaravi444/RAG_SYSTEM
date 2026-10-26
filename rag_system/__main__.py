"""
Entry point for running the FastAPI server directly:

    python -m rag_system
"""

import os
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import uvicorn
from rag_system.config import settings

if __name__ == "__main__":
    print(f"\n>> Starting RAG System Web & API Server on http://{settings.api_host}:{settings.api_port} ...")
    print(f">> Interactive Swagger Docs: http://{settings.api_host}:{settings.api_port}/docs")
    print(f">> Web User Interface:       http://{settings.api_host}:{settings.api_port}/\n")
    uvicorn.run(
        "rag_system.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
