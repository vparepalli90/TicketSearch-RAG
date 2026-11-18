"""
Module 1: Embeddings & Semantic Similarity
==========================================

What you'll learn:
  1. Generate embeddings with the OpenAI API
  2. Compute cosine similarity between texts
  3. Run a basic semantic search over support tickets
  4. Visualize similarity with a heatmap (not a lossy 2D projection)

Run:
  cd modules/1_embeddings
  python demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")  # save images without opening a window
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from openai import OpenAI

from shared.utils import (
    EMBEDDING_MODEL,
    banner,
    load_tickets,
    require_api_key,
    ticket_to_text,
)

OUTPUT_IMAGE = Path(__file__).parent / "embeddings_similarity_analysis.png"


# ---------------------------------------------------------------------------
# 1. Embedding helpers
# ---------------------------------------------------------------------------
def get_embeddings(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed a batch of texts in ONE API call (cheaper and faster than a loop)."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    # Responses come back in the same order as the inputs
    return np.array([item.embedding for item in response.data])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(θ) = (a · b) / (‖a‖ ‖b‖). OpenAI vectors are already unit length,
    so this equals the dot product, but we compute the full formula to teach it."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """All pairwise cosine similarities at once."""
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return normed @ normed.T


# ---------------------------------------------------------------------------
# 2. Demos
# ---------------------------------------------------------------------------
def demo_basic_embedding(client: OpenAI) -> None:
    banner("PART 1: What does an embedding look like?")
    text = "Users cannot log in after a password reset"
    vec = get_embeddings(client, [text])[0]
    print(f"Text      : {text}")
    print(f"Model     : {EMBEDDING_MODEL}")
    print(f"Dimensions: {len(vec)}")
    print(f"First 8   : {np.round(vec[:8], 4)}")
    print(f"L2 norm   : {np.linalg.norm(vec):.4f}  (OpenAI embeddings are normalized)")


def demo_pairwise_similarity(client: OpenAI) -> None:
    banner("PART 2: Semantic similarity between sentences")
    sentences = [
        "Users can't sign in after resetting their password",   # auth
        "Login fails with invalid credentials after reset",     # auth (paraphrase)
        "The nightly ETL job timed out on the orders table",    # data pipeline
        "Data warehouse load is taking too long and failing",   # data pipeline (paraphrase)
        "What's a good recipe for banana bread?",               # unrelated
    ]
    vecs = get_embeddings(client, sentences)
    base = 0
    print(f"Reference: '{sentences[base]}'\n")
    for i, s in enumerate(sentences):
        if i == base:
            continue
        score = cosine_similarity(vecs[base], vecs[i])
        print(f"  {score:.3f}  {s}")
    print("\n👉 Paraphrases score high even with few shared words — that's semantics, not keywords.")


def demo_semantic_search(client: OpenAI, tickets: list[dict], ticket_vecs: np.ndarray) -> None:
    banner("PART 3: Semantic search over support tickets")
    queries = [
        "people stuck in a redirect loop when signing in",
        "warehouse load is too slow",
        "our cloud data bill went way up",
    ]
    query_vecs = get_embeddings(client, queries)
    for q, qv in zip(queries, query_vecs):
        scores = [cosine_similarity(qv, tv) for tv in ticket_vecs]
        top = np.argsort(scores)[::-1][:3]
        print(f"\n🔎 Query: {q}")
        for rank, idx in enumerate(top, 1):
            t = tickets[idx]
            print(f"   {rank}. [{scores[idx]:.3f}] {t['ticket_id']} — {t['title']}")


def demo_heatmap(tickets: list[dict], ticket_vecs: np.ndarray) -> None:
    banner("PART 4: Similarity heatmap")
    sim = cosine_similarity_matrix(ticket_vecs)
    labels = [f"{t['ticket_id']} ({t['category'][:6]})" for t in tickets]

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        sim,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        vmin=float(sim[~np.eye(len(sim), dtype=bool)].min()),
        vmax=1.0,
        square=True,
        cbar_kws={"label": "Cosine similarity"},
        ax=ax,
    )
    ax.set_title("Ticket-to-ticket cosine similarity", fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=120)
    plt.close(fig)
    print(f"Saved heatmap → {OUTPUT_IMAGE.name}")

    # Show the most similar distinct pairs in text, too
    n = len(tickets)
    pairs = [(sim[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(reverse=True)
    print("\nMost similar ticket pairs:")
    for score, i, j in pairs[:5]:
        print(f"  {score:.3f}  {tickets[i]['ticket_id']} ↔ {tickets[j]['ticket_id']}  "
              f"({tickets[i]['category']} / {tickets[j]['category']})")


def main() -> None:
    require_api_key()
    client = OpenAI()
    tickets = load_tickets()
    print(f"Loaded {len(tickets)} tickets")

    demo_basic_embedding(client)
    demo_pairwise_similarity(client)

    # Embed every ticket once and reuse for search + heatmap
    ticket_vecs = get_embeddings(client, [ticket_to_text(t) for t in tickets])
    demo_semantic_search(client, tickets, ticket_vecs)
    demo_heatmap(tickets, ticket_vecs)

    banner("✅ Module 1 complete")
    print("Key takeaway: embeddings place similar meanings close together;")
    print("cosine similarity measures that closeness.")


if __name__ == "__main__":
    main()
