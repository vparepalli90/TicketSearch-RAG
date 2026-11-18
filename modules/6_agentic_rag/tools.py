"""
Custom tools the agent can call.

Each tool is a plain Python function with a docstring — the docstring is
what the LLM reads to decide WHEN to call the tool, so write it carefully.
"""

import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

from shared.utils import get_langchain_embeddings, load_tickets, ticket_metadata, ticket_to_text

VECTORSTORE_DIR = Path(__file__).parent / "agent_vectorstore"


@lru_cache(maxsize=1)
def _tickets() -> dict[str, dict]:
    return {t["ticket_id"]: t for t in load_tickets()}


@lru_cache(maxsize=1)
def _store() -> Chroma:
    """Build the vector store once, then reuse it on later runs."""
    embeddings = get_langchain_embeddings()
    store = Chroma(
        collection_name="agent_tickets",
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    if store._collection.count() == 0:
        docs = [Document(page_content=ticket_to_text(t), metadata=ticket_metadata(t))
                for t in _tickets().values()]
        store.add_documents(docs, ids=[d.metadata["ticket_id"] for d in docs])
    return store


def _fmt(t: dict) -> str:
    return (f"{t['ticket_id']} | {t['title']} | {t['category']} | {t['priority']} | {t['created_date']}\n"
            f"  Problem: {t['description']}\n  Resolution: {t['resolution']}")


@tool
def search_similar_tickets(query: str, k: int = 3) -> str:
    """Semantic search over past support tickets. Use this for any question that
    describes a symptom or problem in natural language (e.g. 'users can't log in',
    'ETL is slow'). Returns the most similar tickets with their resolutions."""
    k = max(1, min(k, 8))
    hits = _store().similarity_search_with_relevance_scores(query, k=k)
    if not hits:
        return "No similar tickets found."
    return "\n\n".join(
        f"[relevance {score:.2f}] {_fmt(_tickets()[d.metadata['ticket_id']])}" for d, score in hits
    )


@tool
def get_ticket_by_id(ticket_id: str) -> str:
    """Fetch one ticket by its exact ID, such as 'TICK-007'. Use this when the user
    mentions a ticket ID or when you need full details of a ticket found earlier."""
    t = _tickets().get(ticket_id.strip().upper())
    return _fmt(t) if t else f"No ticket with ID {ticket_id}."


@tool
def search_by_category(category: str, priority: str = "") -> str:
    """List tickets in a category, optionally filtered by priority.
    Categories: Authentication, Data Pipeline, Reporting, Infrastructure, Database,
    Integration, Network, Performance. Priorities: Low, Medium, High, Critical."""
    cat = category.strip().lower()
    pri = priority.strip().lower()
    matches = [t for t in _tickets().values()
               if t["category"].lower() == cat and (not pri or t["priority"].lower() == pri)]
    if not matches:
        return f"No tickets found for category='{category}' priority='{priority or 'any'}'."
    return "\n\n".join(_fmt(t) for t in matches)


@tool
def get_ticket_statistics() -> str:
    """Counts of tickets by category, priority and status. Use this for questions
    about volume, trends or 'how many' — not for troubleshooting."""
    tickets = list(_tickets().values())
    lines = [f"Total tickets: {len(tickets)}"]
    for field in ("category", "priority", "status"):
        counts = Counter(t[field] for t in tickets).most_common()
        lines.append(f"By {field}: " + ", ".join(f"{k}={v}" for k, v in counts))
    return "\n".join(lines)


ALL_TOOLS = [search_similar_tickets, get_ticket_by_id, search_by_category, get_ticket_statistics]
