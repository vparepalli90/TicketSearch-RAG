"""
Shared helpers used by every module.

Keeps configuration, data loading and ticket formatting in one place so each
module's demo.py can focus on the concept it teaches.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Repo root = parent of this "shared" folder
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "synthetic_tickets.json"

# Load .env from the repo root no matter which module folder we run from
load_dotenv(REPO_ROOT / ".env")

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def require_api_key() -> str:
    """Return the OpenAI API key or exit with a clear message."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("your_") or key.startswith("sk-your"):
        print("❌ OPENAI_API_KEY is not set.")
        print(f"   Copy .env.example to .env in {REPO_ROOT} and add your key.")
        sys.exit(1)
    return key


def get_langchain_embeddings():
    """LangChain OpenAI embeddings configured for this project.

    check_embedding_ctx_length=False skips LangChain's tiktoken pre-check,
    which downloads a tokenizer file at runtime (often blocked on corporate
    networks). Our tickets are far below the 8k-token embedding limit.
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=EMBEDDING_MODEL, check_embedding_ctx_length=False)


def load_tickets(path: Path = DATA_PATH) -> list[dict]:
    """Load the synthetic support tickets."""
    if not path.exists():
        print(f"❌ Ticket file not found: {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ticket_to_text(ticket: dict) -> str:
    """Flatten one ticket into a single text block for embedding/retrieval."""
    return (
        f"Ticket ID: {ticket['ticket_id']}\n"
        f"Title: {ticket['title']}\n"
        f"Category: {ticket['category']} | Priority: {ticket['priority']} | "
        f"Status: {ticket['status']}\n"
        f"Description: {ticket['description']}\n"
        f"Resolution: {ticket['resolution']}\n"
        f"Tags: {', '.join(ticket.get('tags', []))}"
    )


def ticket_metadata(ticket: dict) -> dict:
    """Metadata kept alongside each document (must be flat scalars for Chroma)."""
    return {
        "ticket_id": ticket["ticket_id"],
        "title": ticket["title"],
        "category": ticket["category"],
        "priority": ticket["priority"],
        "status": ticket["status"],
        "created_date": ticket["created_date"],
    }


def banner(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
