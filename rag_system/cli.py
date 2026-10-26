"""
CLI interface using Click + Rich for beautiful terminal output.

Usage:
    rag ingest <path>       — Ingest a file or directory
    rag ask "<question>"    — Ask a question
    rag chat                — Interactive chat mode
    rag status              — Show vector store stats
    rag reset               — Clear all ingested documents
"""

from __future__ import annotations

import os
import sys

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

custom_theme = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "question": "bold magenta",
    }
)

console = Console(theme=custom_theme)


def _print_banner():
    """Display the application banner."""
    banner = """
╔══════════════════════════════════════════════════╗
║          🧠  RAG System v1.0                     ║
║   Groq • ChromaDB • LangChain                   ║
╚══════════════════════════════════════════════════╝"""
    console.print(banner, style="bold cyan")


@click.group()
@click.version_option(version="1.0.0", prog_name="RAG System")
def cli():
    """🧠 RAG System — Retrieval-Augmented Generation CLI."""
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--recursive", "-r", is_flag=True, default=True, help="Recursively scan directories.")
def ingest(path: str, recursive: bool):
    """📥 Ingest documents from a file or directory."""
    from pathlib import Path as P

    from rag_system.pipeline import ingest_directory, ingest_file

    _print_banner()
    target = P(path)

    try:
        with console.status("[bold cyan]Ingesting documents...", spinner="dots"):
            if target.is_dir():
                result = ingest_directory(target)
            else:
                result = ingest_file(target)

        # Display results
        table = Table(title="✅ Ingestion Complete", show_header=False, border_style="green")
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")

        if "file" in result:
            table.add_row("File", result["file"])
        else:
            table.add_row("Directory", result["directory"])

        table.add_row("Documents loaded", str(result["documents_loaded"]))
        table.add_row("Chunks created", str(result["chunks_created"]))
        table.add_row("Chunks stored", str(result["chunks_added"]))
        console.print(table)

    except Exception as exc:
        console.print(f"\n[error]❌ Error: {exc}[/error]")
        sys.exit(1)


@cli.command()
@click.argument("question")
@click.option("--top-k", "-k", default=None, type=int, help="Number of chunks to retrieve.")
@click.option("--sources", "-s", is_flag=True, help="Show source documents.")
def ask(question: str, top_k: int | None, sources: bool):
    """❓ Ask a question against the knowledge base."""
    from rag_system.pipeline import ask as pipeline_ask

    _print_banner()

    try:
        with console.status("[bold cyan]Thinking...", spinner="dots"):
            result = pipeline_ask(question, top_k=top_k, return_sources=sources)

        # Display answer
        console.print()
        console.print(
            Panel(
                Markdown(result["answer"]),
                title="🤖 Answer",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Display sources if requested
        if sources and result.get("sources"):
            console.print()
            table = Table(title="📄 Sources", border_style="blue")
            table.add_column("#", style="dim", width=3)
            table.add_column("Source", style="cyan")
            table.add_column("Score", style="yellow", width=8)
            table.add_column("Preview", max_width=60)

            for i, src in enumerate(result["sources"], 1):
                source = src["metadata"].get("source", "unknown")
                score = str(src.get("score", ""))
                preview = src["content"][:80].replace("\n", " ") + "..."
                table.add_row(str(i), source, score, preview)

            console.print(table)

    except Exception as exc:
        console.print(f"\n[error]❌ Error: {exc}[/error]")
        sys.exit(1)


@cli.command()
@click.option("--top-k", "-k", default=None, type=int, help="Number of chunks to retrieve.")
def chat(top_k: int | None):
    """💬 Start an interactive chat session."""
    from rag_system.pipeline import ask as pipeline_ask

    _print_banner()
    console.print("\n[info]Type your questions below. Type 'quit' or 'exit' to stop.[/info]\n")

    while True:
        try:
            question = console.input("[question]You:[/question] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[info]Goodbye! 👋[/info]")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            console.print("[info]Goodbye! 👋[/info]")
            break

        try:
            with console.status("[bold cyan]Thinking...", spinner="dots"):
                result = pipeline_ask(question, top_k=top_k, return_sources=False)

            console.print()
            console.print(
                Panel(
                    Markdown(result["answer"]),
                    title="🤖 Answer",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            console.print()

        except Exception as exc:
            console.print(f"[error]❌ Error: {exc}[/error]\n")


@cli.command()
def status():
    """📊 Show vector store statistics."""
    from rag_system.pipeline import status as pipeline_status

    _print_banner()

    try:
        stats = pipeline_status()

        table = Table(title="📊 Vector Store Status", show_header=False, border_style="cyan")
        table.add_column("Property", style="bold")
        table.add_column("Value", style="cyan")
        table.add_row("Collection", stats["collection_name"])
        table.add_row("Documents", str(stats["document_count"]))
        table.add_row("Storage", stats["persist_directory"])
        console.print(table)

    except Exception as exc:
        console.print(f"\n[error]❌ Error: {exc}[/error]")
        sys.exit(1)


@cli.command()
@click.confirmation_option(prompt="⚠️  This will delete ALL ingested documents. Continue?")
def reset():
    """🗑️  Clear all documents from the vector store."""
    from rag_system.pipeline import reset as pipeline_reset

    _print_banner()

    try:
        result = pipeline_reset()
        console.print(f"\n[success]✅ {result['message']}[/success]")
    except Exception as exc:
        console.print(f"\n[error]❌ Error: {exc}[/error]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
