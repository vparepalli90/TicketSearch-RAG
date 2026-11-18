"""
Module 2: Chunking Strategies
=============================

What you'll learn:
  1. Fixed-size chunking (CharacterTextSplitter)
  2. Recursive chunking (RecursiveCharacterTextSplitter) — best general default
  3. Semantic chunking (split where meaning shifts)
  4. Structure-aware splitting for Markdown and HTML
  5. Build and query a Chroma vector store

Run:
  cd modules/2_chunking
  python demo.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from shared.utils import (
    get_langchain_embeddings,
    banner,
    load_tickets,
    require_api_key,
    ticket_metadata,
    ticket_to_text,
)

CHROMA_DIR = Path(__file__).parent / "chroma_db"

# A longer runbook gives the splitters something interesting to cut.
RUNBOOK_MD = """# Incident Runbook: Authentication Failures

## Overview
This runbook covers login failures, SSO redirect loops and MFA lockouts.
Always check the status page first. If more than 5% of logins fail, page the on-call identity engineer.

## Step 1: Triage
Confirm the scope. Is it all users or a subset? Check the IdP dashboard for error spikes.
Look at recent changes: patches, config deploys, certificate rotations.

### Common signals
- "Invalid credentials" right after password reset usually means a stale credential cache.
- Redirect loops between the app and the IdP usually mean the session cookie is dropped.
- MFA failures after a phone change mean the user needs re-enrollment.

## Step 2: Fix credential cache issues
Flush the credential cache on every auth node. Verify the cache TTL in idp-config.yaml is 5 minutes.
Ask an affected user to retry. If it still fails, collect a HAR file.

## Step 3: Fix SSO redirect loops
Check the session cookie attributes. SameSite=Strict breaks cross-site redirects from the IdP.
Use SameSite=Lax with the Secure flag. Clear the user's cookies and retry.

## Step 4: MFA lockouts
Verify identity through the user's manager. Issue a one-time bypass code valid for 1 hour.
Have the user re-enroll and generate backup codes.

## Escalation
If none of the above works within 30 minutes, escalate to the Identity Platform team in the #identity-oncall channel.
"""

RUNBOOK_HTML = """
<html><body>
<h1>Database Performance Runbook</h1>
<p>Use this when SQL Server CPU is high or queries slow down.</p>
<h2>Check Query Store</h2>
<p>Find the top queries by CPU in the last hour. Look for plan regressions.</p>
<h2>Parameter sniffing</h2>
<p>If one plan is good for small parameters and bad for large ones, try OPTION (RECOMPILE) or a plan guide.</p>
<h2>Deadlocks</h2>
<p>Error 1205 means a deadlock. Standardize lock ordering and add retry with backoff.</p>
</body></html>
"""


def show_chunks(name: str, chunks: list, max_show: int = 3) -> None:
    sizes = [len(c.page_content if hasattr(c, "page_content") else c) for c in chunks]
    print(f"\n▶ {name}: {len(chunks)} chunks | avg {sum(sizes) // max(len(sizes), 1)} chars "
          f"| min {min(sizes)} | max {max(sizes)}")
    for i, c in enumerate(chunks[:max_show], 1):
        text = c.page_content if hasattr(c, "page_content") else c
        meta = getattr(c, "metadata", {})
        preview = text.replace("\n", " ")[:110]
        print(f"   [{i}] {preview}...")
        if meta:
            print(f"       metadata: {meta}")


def demo_fixed_size() -> None:
    banner("PART 1: Fixed-size chunking")
    splitter = CharacterTextSplitter(separator=" ", chunk_size=300, chunk_overlap=30)
    chunks = splitter.split_text(RUNBOOK_MD)
    show_chunks("CharacterTextSplitter (300 chars, 30 overlap)", chunks)
    print("\n⚠️  Notice: chunks cut mid-sentence and ignore section boundaries.")


def demo_recursive() -> None:
    banner("PART 2: Recursive chunking (recommended default)")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,  # ~10% overlap keeps context across boundaries
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(RUNBOOK_MD)
    show_chunks("RecursiveCharacterTextSplitter (300 chars, 30 overlap)", chunks)
    print("\n👉 It tries big separators first (sections, paragraphs) and only")
    print("   falls back to smaller ones when a piece is still too long.")


def demo_semantic(embeddings: OpenAIEmbeddings) -> None:
    banner("PART 3: Semantic chunking")
    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile",
                              breakpoint_threshold_amount=80)
    chunks = chunker.create_documents([RUNBOOK_MD])
    show_chunks("SemanticChunker (split where meaning shifts)", chunks)
    print("\n👉 Costs an embedding call per sentence, but chunk edges follow topic changes.")


def demo_structure_aware() -> None:
    banner("PART 4: Structure-aware splitting (Markdown & HTML)")
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "doc"), ("##", "section"), ("###", "subsection")]
    )
    md_chunks = md_splitter.split_text(RUNBOOK_MD)
    show_chunks("MarkdownHeaderTextSplitter", md_chunks, max_show=4)

    html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "doc"), ("h2", "section")])
    html_chunks = html_splitter.split_text(RUNBOOK_HTML)
    show_chunks("HTMLHeaderTextSplitter", html_chunks, max_show=4)
    print("\n👉 Headers become metadata — great for filtering and citations.")


def demo_vector_store(embeddings: OpenAIEmbeddings) -> None:
    banner("PART 5: Build a Chroma vector store from tickets + runbook")
    tickets = load_tickets()

    # Tickets are short, so each ticket is one document
    docs = [Document(page_content=ticket_to_text(t), metadata=ticket_metadata(t)) for t in tickets]

    # The runbook is long, so split it by headers then recursively
    md_sections = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "doc"), ("##", "section")]
    ).split_text(RUNBOOK_MD)
    runbook_chunks = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40).split_documents(md_sections)
    for c in runbook_chunks:
        c.metadata["source"] = "auth_runbook"
        c.metadata.setdefault("ticket_id", "RUNBOOK")
    docs.extend(runbook_chunks)
    print(f"Indexing {len(docs)} documents ({len(tickets)} tickets + {len(runbook_chunks)} runbook chunks)")

    # Start fresh each run so results are reproducible
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="tickets",
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    for query in ["how do I fix an SSO redirect loop?", "warehouse ETL keeps timing out"]:
        print(f"\n🔎 Query: {query}")
        for doc, score in store.similarity_search_with_relevance_scores(query, k=3):
            label = doc.metadata.get("ticket_id")
            title = doc.metadata.get("title") or doc.metadata.get("section", "")
            print(f"   [{score:.3f}] {label} — {title}")

    # Metadata filtering
    print("\n🔎 Same query, filtered to category='Authentication':")
    for doc in store.similarity_search("users can't get in", k=3, filter={"category": "Authentication"}):
        print(f"   {doc.metadata['ticket_id']} — {doc.metadata['title']}")

    print(f"\nVector store saved to {CHROMA_DIR.name}/")


def main() -> None:
    require_api_key()
    embeddings = get_langchain_embeddings()

    demo_fixed_size()
    demo_recursive()
    demo_semantic(embeddings)
    demo_structure_aware()
    demo_vector_store(embeddings)

    banner("✅ Module 2 complete")
    print("Key takeaway: bad chunking = bad retrieval. Start with recursive splitting,")
    print("use structure-aware splitters when documents have headers.")


if __name__ == "__main__":
    main()
