"""
FastAPI REST API and Web UI for the RAG system.

Endpoints:
    GET    /                    — Interactive Web Application UI
    GET    /health              — Health check
    POST   /api/ingest          — Admin-key-protected document upload
    POST   /api/ingest/directory — Admin-key-protected server-side ingestion
    POST   /api/query            — Ask a question
    GET    /api/status           — Collection statistics
    DELETE /api/reset            — Admin-key-protected collection reset
    GET    /docs                — Interactive OpenAPI / Swagger documentation
"""

from __future__ import annotations

import hmac
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag_system.config import settings
from rag_system.document_loader import SUPPORTED_EXTENSIONS
from rag_system.logger import get_logger
from rag_system.pipeline import (
    ask,
    ingest_directory,
    ingest_file,
    reset,
    status,
)

logger = get_logger(__name__)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024

# ────────────────────────────────────────────────────────────
# App
# ────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Intelligence System",
    description="Retrieval-Augmented Generation powered by Groq, ChromaDB & LangChain.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ────────────────────────────────────────────────────────────
# Request / Response schemas
# ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve.")
    return_sources: bool = Field(False, description="Include source documents in response.")


class SourceDoc(BaseModel):
    content: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: Optional[List[SourceDoc]] = None


class IngestResponse(BaseModel):
    file: Optional[str] = None
    directory: Optional[str] = None
    documents_loaded: int
    chunks_created: int
    chunks_added: int


class DirectoryIngestRequest(BaseModel):
    path: str = Field(..., description="Absolute path to the directory on the server.")


class StatusResponse(BaseModel):
    collection_name: str
    document_count: int
    persist_directory: str


class ResetResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    embedding_model: str


def require_admin_key(x_admin_key: Optional[str] = Header(default=None)) -> None:
    """Protect document and destructive actions with the configured admin key."""
    configured_key = settings.admin_api_key
    if not configured_key:
        raise HTTPException(
            status_code=503,
            detail="Admin actions are disabled until ADMIN_API_KEY is configured.",
        )
    if not x_admin_key or not hmac.compare_digest(
        x_admin_key.encode("utf-8"), configured_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="A valid X-Admin-Key is required.")


# ────────────────────────────────────────────────────────────
# Interactive Web UI (HTML / Vanilla CSS / Modern JS)
# ────────────────────────────────────────────────────────────

WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Intelligence Studio</title>
    <style>
        :root {
            --bg-base: #090d16;
            --bg-card: rgba(19, 26, 42, 0.75);
            --bg-card-hover: rgba(26, 36, 58, 0.9);
            --border-card: rgba(255, 255, 255, 0.08);
            --border-glow: rgba(99, 102, 241, 0.35);
            --primary: #6366f1;
            --primary-hover: #4f46e5;
            --primary-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --accent-emerald: #10b981;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --radius-lg: 16px;
            --radius-md: 12px;
            --radius-sm: 8px;
            --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(at 10% 20%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
                radial-gradient(at 90% 80%, rgba(168, 85, 247, 0.12) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        header {
            border-bottom: 1px solid var(--border-card);
            background: rgba(9, 13, 22, 0.8);
            backdrop-filter: blur(12px);
            padding: 1rem 2rem;
            position: sticky;
            top: 0;
            z-index: 50;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            text-decoration: none;
            color: inherit;
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            background: var(--primary-gradient);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .brand-badge {
            font-size: 0.7rem;
            padding: 0.2rem 0.5rem;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #818cf8;
            border-radius: 999px;
            font-weight: 600;
        }

        .header-actions {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        .btn-link {
            text-decoration: none;
            font-size: 0.875rem;
            color: var(--text-muted);
            transition: var(--transition);
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        .btn-link:hover {
            color: var(--text-main);
        }

        main {
            flex: 1;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            padding: 2.5rem 1.5rem;
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 2rem;
        }

        @media (max-width: 960px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            backdrop-filter: blur(16px);
            transition: var(--transition);
        }

        .card:hover {
            border-color: var(--border-glow);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1.25rem;
        }

        .card-title {
            font-size: 1.05rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Upload Area */
        .dropzone {
            border: 2px dashed rgba(255, 255, 255, 0.15);
            border-radius: var(--radius-md);
            padding: 2rem 1rem;
            text-align: center;
            cursor: pointer;
            transition: var(--transition);
            background: rgba(255, 255, 255, 0.02);
        }

        .dropzone:hover, .dropzone.dragover {
            border-color: var(--primary);
            background: rgba(99, 102, 241, 0.05);
        }

        .dropzone-icon {
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
        }

        .dropzone-text {
            font-size: 0.875rem;
            color: var(--text-muted);
        }

        .dropzone-subtext {
            font-size: 0.75rem;
            color: rgba(255, 255, 255, 0.4);
            margin-top: 0.35rem;
        }

        /* Stats List */
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            margin-top: 1.25rem;
        }

        .stat-item {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: var(--radius-sm);
            padding: 0.75rem;
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }

        .stat-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-main);
        }

        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.75rem 1.25rem;
            font-size: 0.875rem;
            font-weight: 600;
            border-radius: var(--radius-md);
            border: none;
            cursor: pointer;
            transition: var(--transition);
            width: 100%;
        }

        .btn-primary {
            background: var(--primary-gradient);
            color: white;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
        }

        .btn-danger {
            background: rgba(239, 68, 68, 0.1);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.2);
            margin-top: 0.75rem;
        }

        .btn-danger:hover {
            background: rgba(239, 68, 68, 0.2);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        /* Chat / Q&A Box */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .query-box {
            position: relative;
        }

        .query-input {
            width: 100%;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-md);
            padding: 1rem 4.5rem 1rem 1.25rem;
            color: white;
            font-size: 1rem;
            font-family: inherit;
            outline: none;
            transition: var(--transition);
        }

        .query-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
            background: rgba(255, 255, 255, 0.06);
        }

        .query-submit {
            position: absolute;
            right: 8px;
            top: 8px;
            bottom: 8px;
            width: 44px;
            border-radius: var(--radius-sm);
            background: var(--primary-gradient);
            border: none;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
        }

        .query-submit:hover:not(:disabled) {
            transform: scale(1.05);
        }

        .options-row {
            display: flex;
            align-items: center;
            gap: 1.5rem;
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.5rem;
        }

        .custom-checkbox {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            cursor: pointer;
        }

        /* Results Display */
        .response-card {
            background: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: var(--radius-lg);
            padding: 1.75rem;
            display: none;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .response-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-card);
        }

        .response-content {
            font-size: 1rem;
            line-height: 1.7;
            color: #e2e8f0;
            white-space: pre-wrap;
        }

        .sources-section {
            margin-top: 1.5rem;
            padding-top: 1.25rem;
            border-top: 1px dashed rgba(255, 255, 255, 0.1);
        }

        .sources-title {
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
        }

        .source-pill {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: var(--radius-sm);
            padding: 0.65rem 0.85rem;
            margin-bottom: 0.5rem;
            font-size: 0.825rem;
        }

        .source-meta {
            color: #818cf8;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            margin-bottom: 0.25rem;
        }

        .toast {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            padding: 1rem 1.5rem;
            border-radius: var(--radius-md);
            background: #1e293b;
            color: white;
            font-size: 0.875rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.4);
            border: 1px solid var(--border-card);
            z-index: 100;
            display: none;
            animation: slideUp 0.3s ease;
        }

        @keyframes slideUp {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }

        .spinner {
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 2px solid white;
            width: 18px;
            height: 18px;
            animation: spin 0.8s linear infinite;
            display: inline-block;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        body {
            background-color: #080b12;
            background-image:
                radial-gradient(ellipse at 12% 0%, rgba(79, 70, 229, 0.18), transparent 38%),
                radial-gradient(ellipse at 88% 35%, rgba(14, 165, 233, 0.08), transparent 32%);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        header {
            padding: 1rem clamp(1rem, 4vw, 3rem);
            border-color: rgba(148, 163, 184, 0.12);
        }

        main {
            max-width: 1320px;
            padding: clamp(1.25rem, 4vw, 3.5rem) clamp(1rem, 4vw, 3rem);
            gap: 1.25rem;
            align-content: start;
        }

        .intro {
            grid-column: 1 / -1;
            padding: 0.5rem 0 0.35rem;
        }

        .intro-kicker {
            color: #a5b4fc;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
        }

        .intro h1 {
            margin-top: 0.65rem;
            font-size: clamp(2rem, 4vw, 3.25rem);
            line-height: 1.1;
            letter-spacing: -0.055em;
        }

        .intro h1 span {
            color: #a5b4fc;
        }

        .intro p {
            max-width: 650px;
            margin-top: 0.8rem;
            color: var(--text-muted);
            line-height: 1.7;
        }

        .card, .response-card {
            background: rgba(15, 22, 36, 0.82);
            border-color: rgba(148, 163, 184, 0.16);
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.16);
        }

        .card:hover {
            border-color: rgba(129, 140, 248, 0.38);
        }

        .dropzone {
            border-width: 1px;
            padding: 1.75rem 1rem;
        }

        .dropzone:focus-within, .dropzone:focus-visible {
            outline: 3px solid rgba(129, 140, 248, 0.5);
            outline-offset: 3px;
        }

        .admin-key-label {
            display: block;
            margin: 1rem 0 0.4rem;
            color: var(--text-muted);
            font-size: 0.78rem;
            font-weight: 600;
        }

        .admin-key-input {
            width: 100%;
            border: 1px solid var(--border-card);
            border-radius: var(--radius-sm);
            padding: 0.7rem 0.8rem;
            background: rgba(255, 255, 255, 0.04);
            color: var(--text-main);
            font: inherit;
        }

        .admin-key-input:focus, .query-input:focus, select:focus {
            outline: 3px solid rgba(129, 140, 248, 0.38);
            outline-offset: 2px;
        }

        .admin-key-help {
            margin-top: 0.4rem;
            color: #718096;
            font-size: 0.72rem;
            line-height: 1.5;
        }

        .prompt-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .prompt-chip {
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 999px;
            padding: 0.5rem 0.75rem;
            background: rgba(255, 255, 255, 0.035);
            color: #cbd5e1;
            font: inherit;
            font-size: 0.76rem;
            cursor: pointer;
            transition: var(--transition);
        }

        .prompt-chip:hover {
            border-color: rgba(129, 140, 248, 0.6);
            color: white;
        }

        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--text-muted);
            font-size: 0.78rem;
        }

        .status-indicator::before {
            width: 0.5rem;
            height: 0.5rem;
            border-radius: 50%;
            background: #64748b;
            content: "";
        }

        .status-indicator.online::before { background: #34d399; }
        .status-indicator.offline::before { background: #fb7185; }

        .response-content {
            overflow-wrap: anywhere;
        }

        button:focus-visible, a:focus-visible, input:focus-visible, select:focus-visible {
            outline: 3px solid rgba(129, 140, 248, 0.6);
            outline-offset: 3px;
        }

        .visually-hidden {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }

        @media (max-width: 640px) {
            header { padding: 0.85rem 1rem; }
            main { gap: 1rem; }
            .intro h1 { font-size: 2.15rem; }
            .card, .response-card { padding: 1.15rem; }
            .options-row { align-items: flex-start; flex-direction: column; gap: 0.8rem; }
            .header-actions .btn-link { font-size: 0; }
            .header-actions .btn-link svg { width: 20px; height: 20px; }
        }
    </style>
</head>
<body>

    <header>
        <a href="/" class="brand">
            <div class="brand-icon">⚡</div>
            <div>
                <div class="brand-title">RAG Studio</div>
            </div>
            <span class="brand-badge" id="model-badge">Groq</span>
        </a>
        <div class="header-actions">
            <a href="/docs" target="_blank" class="btn-link">
                <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/></svg>
                Swagger API Docs
            </a>
        </div>
    </header>

    <main>
        <section class="intro" aria-labelledby="page-title">
            <div class="intro-kicker">Document-grounded knowledge assistant</div>
            <h1 id="page-title">Make your documents <span>answer back.</span></h1>
            <p>Bring your files together, then ask clear questions and get answers grounded in your sources. Anyone with this site URL can query indexed content, so do not upload confidential documents.</p>
        </section>
        <!-- Sidebar: Ingest & Stats -->
        <aside style="display: flex; flex-direction: column; gap: 1.5rem;">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">📁 Document Ingestion</h2>
                </div>
                
                <label class="dropzone" id="dropzone" for="file-input" role="button" tabindex="0" aria-label="Choose a document to upload">
                    <div class="dropzone-icon">📤</div>
                    <div class="dropzone-text">Choose a file or drag it here</div>
                    <div class="dropzone-subtext">PDF, TXT, DOCX, MD, CSV, HTML · up to 10 MB</div>
                </label>
                <input class="visually-hidden" type="file" id="file-input" accept=".pdf,.txt,.docx,.md,.csv,.html,.htm" onchange="handleFileSelect(event)">

                <div id="file-details" style="display: none; margin-top: 1rem; font-size: 0.85rem; color: var(--text-muted);">
                    Selected: <span id="file-name" style="color: white; font-weight: 600;"></span>
                    <button class="btn btn-primary" id="btn-upload" style="margin-top: 0.75rem;" onclick="uploadFile()">
                        Index Document
                    </button>
                </div>
            </div>

            <div class="card">
                <label class="admin-key-label" for="admin-key">Admin key</label>
                <input class="admin-key-input" type="password" id="admin-key" autocomplete="off" placeholder="Required to upload or reset">
                <div class="admin-key-help">Enter the server's ADMIN_API_KEY. It is used only in this browser session and is not saved.</div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">📊 Knowledge Store</h2>
                    <span class="status-indicator" id="connection-state" role="status">Connecting</span>
                </div>

                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-label">Indexed chunks</div>
                        <div class="stat-value" id="stat-docs">-</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Vector Collection</div>
                        <div class="stat-value" id="stat-col" style="font-size: 0.9rem; word-break: break-all;">-</div>
                    </div>
                </div>

                <button class="btn btn-danger" onclick="resetDatabase()">
                    🗑️ Reset Knowledge Base
                </button>
            </div>
        </aside>

        <!-- Main Query Section -->
        <section class="chat-container">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">💬 Ask Your Documents</h2>
                </div>

                <div class="query-box">
                    <input 
                        type="text" 
                        class="query-input" 
                        id="query-input" 
                        placeholder="e.g. Summarize the key findings or ask any question..." 
                        onkeydown="if(event.key === 'Enter') runQuery()"
                    >
                    <button class="query-submit" id="btn-query" onclick="runQuery()">
                        <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>
                    </button>
                </div>

                <div class="options-row">
                    <label class="custom-checkbox">
                        <input type="checkbox" id="opt-sources" checked> Show Source Citations
                    </label>
                    <div style="display: flex; align-items: center; gap: 0.4rem;">
                        Top Chunks:
                        <select id="opt-k" style="background: rgba(255,255,255,0.05); color: white; border: 1px solid var(--border-card); border-radius: 4px; padding: 2px 6px;">
                            <option value="3">3</option>
                            <option value="5" selected>5</option>
                            <option value="10">10</option>
                        </select>
                    </div>
                </div>
                <div class="prompt-chips" aria-label="Example questions">
                    <button class="prompt-chip" type="button" onclick="usePrompt(this)">Summarize the key ideas</button>
                    <button class="prompt-chip" type="button" onclick="usePrompt(this)">What should I know first?</button>
                    <button class="prompt-chip" type="button" onclick="usePrompt(this)">Compare the main findings</button>
                </div>
            </div>

            <!-- Answer display -->
            <div class="response-card" id="response-card">
                <div class="response-header">
                    <div style="font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
                        <span>🤖</span> Synthesized Answer
                    </div>
                    <span id="response-time" style="font-size: 0.75rem; color: var(--text-muted);"></span>
                </div>
                <div class="response-content" id="response-text"></div>

                <div class="sources-section" id="sources-container" style="display: none;">
                    <div class="sources-title">Retrieved Reference Chunks</div>
                    <div id="sources-list"></div>
                </div>
            </div>
        </section>
    </main>

    <div class="toast" id="toast" role="status" aria-live="polite"></div>

    <script>
        let selectedFile = null;
        let toastTimer = null;

        function showToast(msg, isError = false) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.style.borderColor = isError ? '#ef4444' : '#10b981';
            t.style.display = 'block';
            clearTimeout(toastTimer);
            toastTimer = setTimeout(() => { t.style.display = 'none'; }, 4000);
        }

        async function fetchStatus() {
            const state = document.getElementById('connection-state');
            try {
                const res = await fetch('/api/status');
                if (!res.ok) throw new Error('Status request failed');
                const data = await res.json();
                document.getElementById('stat-docs').textContent = data.document_count.toLocaleString();
                document.getElementById('stat-col').textContent = data.collection_name;
                state.textContent = 'Connected';
                state.className = 'status-indicator online';
            } catch (e) {
                state.textContent = 'Unavailable';
                state.className = 'status-indicator offline';
                showToast('Could not connect to the knowledge store.', true);
            }
        }

        async function fetchHealth() {
            try {
                const res = await fetch('/health');
                if (!res.ok) throw new Error('Health request failed');
                const data = await res.json();
                document.getElementById('model-badge').textContent = data.model;
            } catch (e) {
                document.getElementById('model-badge').textContent = 'Offline';
            }
        }

        function getAdminHeaders() {
            const key = document.getElementById('admin-key').value.trim();
            if (!key) {
                showToast('Enter the admin key to manage documents.', true);
                return null;
            }
            return { 'X-Admin-Key': key };
        }

        function usePrompt(button) {
            const input = document.getElementById('query-input');
            input.value = button.textContent;
            input.focus();
        }

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            if (file.size > 10 * 1024 * 1024) {
                selectedFile = null;
                document.getElementById('file-input').value = '';
                showToast('This file is larger than the 10 MB upload limit.', true);
                return;
            }
            selectedFile = file;
            document.getElementById('file-name').textContent =
                `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
            document.getElementById('file-details').style.display = 'block';
        }

        async function uploadFile() {
            if (!selectedFile) return;
            const adminHeaders = getAdminHeaders();
            if (!adminHeaders) return;
            const btn = document.getElementById('btn-upload');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Processing...';

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const res = await fetch('/api/ingest', {
                    method: 'POST',
                    headers: adminHeaders,
                    body: formData
                });
                const data = await res.json();
                if (res.ok) {
                    showToast(`Indexed ${data.chunks_added} chunks from ${selectedFile.name}.`);
                    selectedFile = null;
                    document.getElementById('file-input').value = '';
                    document.getElementById('file-details').style.display = 'none';
                    fetchStatus();
                } else {
                    showToast(data.detail || 'Upload failed.', true);
                }
            } catch (err) {
                showToast('Could not upload the document. Check your connection and retry.', true);
            } finally {
                btn.disabled = false;
                btn.innerText = 'Index Document';
            }
        }

        async function runQuery() {
            const input = document.getElementById('query-input');
            const q = input.value.trim();
            if (!q) return;

            const btn = document.getElementById('btn-query');
            const responseCard = document.getElementById('response-card');
            const responseText = document.getElementById('response-text');
            const sourcesContainer = document.getElementById('sources-container');
            const sourcesList = document.getElementById('sources-list');

            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span>';
            responseCard.style.display = 'block';
            responseText.textContent = 'Searching your documents and preparing an answer...';
            sourcesContainer.style.display = 'none';

            const startTime = performance.now();

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: q,
                        top_k: parseInt(document.getElementById('opt-k').value),
                        return_sources: document.getElementById('opt-sources').checked
                    })
                });

                const data = await res.json();
                const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
                document.getElementById('response-time').textContent = `Answered in ${elapsed}s`;

                if (res.ok) {
                    responseText.textContent = data.answer;

                    if (data.sources && data.sources.length > 0) {
                        sourcesList.replaceChildren();
                        data.sources.forEach((source, idx) => {
                            const item = document.createElement('div');
                            item.className = 'source-pill';
                            const metadata = document.createElement('div');
                            metadata.className = 'source-meta';
                            const sourcePath = source.metadata.source || 'unknown';
                            const sourceName = sourcePath.split('/').pop();
                            metadata.textContent = `Source ${idx + 1} · ${sourceName}`;
                            const content = document.createElement('div');
                            content.style.color = '#cbd5e1';
                            content.style.fontSize = '0.8rem';
                            content.textContent = source.content;
                            item.append(metadata, content);
                            sourcesList.append(item);
                        });
                        sourcesContainer.style.display = 'block';
                    }
                } else {
                    responseText.textContent = data.detail || 'The request failed.';
                }
            } catch (e) {
                responseText.textContent = 'Could not reach the server. Check your connection and try again.';
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>';
            }
        }

        async function resetDatabase() {
            if (!confirm("Are you sure you want to clear all indexed documents from ChromaDB?")) return;
            const adminHeaders = getAdminHeaders();
            if (!adminHeaders) return;
            try {
                const res = await fetch('/api/reset', { method: 'DELETE', headers: adminHeaders });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || 'Reset failed.');
                showToast(data.message || 'Knowledge base cleared.');
                document.getElementById('response-card').style.display = 'none';
                await fetchStatus();
            } catch (e) {
                showToast(e.message || 'Could not reset the knowledge base.', true);
            }
        }

        // Drag & drop handlers
        const dropzone = document.getElementById('dropzone');
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
        });
        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
        });
        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length > 0) {
                handleFileSelect({ target: { files: e.dataTransfer.files } });
            }
        });

        fetchStatus();
        fetchHealth();
    </script>
</body>
</html>
"""


# ────────────────────────────────────────────────────────────
# Endpoints
# ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
async def root_web_ui():
    """Interactive Web Interface for document ingestion and conversational RAG."""
    return HTMLResponse(content=WEB_UI_HTML)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model=settings.llm_model_name,
        embedding_model=settings.embedding_model_name,
    )


@app.post(
    "/api/ingest",
    response_model=IngestResponse,
    tags=["Ingestion"],
    dependencies=[Depends(require_admin_key)],
)
async def ingest_upload(file: UploadFile = File(...)):
    """
    Upload and ingest a single document.

    Supported formats: PDF, TXT, DOCX, MD, CSV, HTML.
    """
    # Validate extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    try:
        filename = Path((file.filename or "uploaded").replace("\\", "/")).name
        filename = "".join(
            character if character.isalnum() or character in "._- " else "_"
            for character in filename
        )
        if not filename or filename in {".", ".."}:
            filename = f"uploaded{suffix}"

        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir) / filename
            total_size = 0
            with tmp_path.open("wb") as tmp:
                while chunk := await file.read(1024 * 1024):
                    total_size += len(chunk)
                    if total_size > MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="File is too large. The upload limit is 10 MB.",
                        )
                    tmp.write(chunk)

            result = ingest_file(tmp_path)
        result["file"] = file.filename
        return IngestResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from exc

    finally:
        await file.close()


@app.post(
    "/api/ingest/directory",
    response_model=IngestResponse,
    tags=["Ingestion"],
    dependencies=[Depends(require_admin_key)],
)
async def ingest_from_directory(request: DirectoryIngestRequest):
    """
    Ingest all supported documents from a server-side directory.
    """
    try:
        result = ingest_directory(request.path)
        return IngestResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Directory ingest failed")
        raise HTTPException(status_code=500, detail="Directory ingestion failed.") from exc


@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_endpoint(request: QueryRequest):
    """
    Ask a question against the ingested knowledge base.
    """
    try:
        result = ask(
            question=request.question,
            top_k=request.top_k,
            return_sources=request.return_sources,
        )

        sources = None
        if request.return_sources and "sources" in result:
            sources = []
            for source in result["sources"]:
                metadata = dict(source["metadata"])
                source_path = str(metadata.get("source", "unknown")).replace("\\", "/")
                metadata["source"] = Path(source_path).name
                sources.append(
                    SourceDoc(
                        content=source["content"],
                        metadata=metadata,
                        score=source["score"],
                    )
                )

        return QueryResponse(answer=result["answer"], sources=sources)

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail="Query failed. Please try again.") from exc


@app.get("/api/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    """Get vector store statistics."""
    try:
        stats = status()
        return StatusResponse(**stats)
    except Exception as exc:
        logger.exception("Status check failed")
        raise HTTPException(status_code=500, detail="Unable to retrieve knowledge base status.") from exc


@app.delete(
    "/api/reset",
    response_model=ResetResponse,
    tags=["System"],
    dependencies=[Depends(require_admin_key)],
)
async def reset_collection():
    """Delete all documents from the vector store."""
    try:
        result = reset()
        return ResetResponse(**result)
    except Exception as exc:
        logger.exception("Reset failed")
        raise HTTPException(status_code=500, detail="Unable to reset the knowledge base.") from exc
