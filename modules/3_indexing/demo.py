"""
Module 3: Indexing Strategies with LlamaIndex
=============================================

Compares five ways to index the same tickets:
  1. Vector Index        — embedding similarity (the usual default)
  2. Summary Index       — scans every node; good for "summarize everything"
  3. Tree Index          — LLM-built hierarchy of summaries, walked top-down
  4. Keyword Table Index — classic keyword → node lookup
  5. Hybrid Retrieval    — BM25 keywords + vectors, merged with Reciprocal Rank Fusion

Run:
  cd modules/3_indexing
  python demo.py
"""

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llama_index.core import (
    Document,
    Settings,
    SimpleKeywordTableIndex,
    SummaryIndex,
    TreeIndex,
    VectorStoreIndex,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from rank_bm25 import BM25Okapi

from shared.utils import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    banner,
    load_tickets,
    require_api_key,
    ticket_metadata,
    ticket_to_text,
)

TEST_QUERIES = [
    "SSO redirect loop with Okta",                     # semantic + keyword overlap
    "error 1205",                                      # exact code — keywords shine
    "why did our cloud data warehouse bill go up?",    # paraphrase — vectors shine
]


# ---------------------------------------------------------------------------
# Hybrid retriever: BM25 + vector with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridRetriever(BaseRetriever):
    """Runs BM25 and vector retrieval, then fuses ranks with RRF.

    RRF score = Σ 1 / (k + rank). It ignores raw score scales, which is
    why it works well for mixing BM25 scores with cosine similarities.
    """

    def __init__(self, vector_retriever, nodes, top_k: int = 3, rrf_k: int = 60):
        super().__init__()
        self.vector_retriever = vector_retriever
        self.nodes = nodes
        self.top_k = top_k
        self.rrf_k = rrf_k
        self.bm25 = BM25Okapi([tokenize(n.get_content()) for n in nodes])

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # Keyword side
        bm25_scores = self.bm25.get_scores(tokenize(query_bundle.query_str))
        bm25_ranked = sorted(range(len(self.nodes)), key=lambda i: bm25_scores[i], reverse=True)

        # Semantic side
        vector_hits = self.vector_retriever.retrieve(query_bundle)

        fused: dict[str, float] = {}
        node_by_id = {n.node_id: n for n in self.nodes}
        for rank, idx in enumerate(bm25_ranked[:10], 1):
            nid = self.nodes[idx].node_id
            fused[nid] = fused.get(nid, 0) + 1 / (self.rrf_k + rank)
        for rank, hit in enumerate(vector_hits, 1):
            nid = hit.node.node_id
            node_by_id.setdefault(nid, hit.node)
            fused[nid] = fused.get(nid, 0) + 1 / (self.rrf_k + rank)

        best = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: self.top_k]
        return [NodeWithScore(node=node_by_id[nid], score=score) for nid, score in best]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def print_hits(hits: list[NodeWithScore]) -> None:
    if not hits:
        print("     (no results)")
    for h in hits:
        tid = h.node.metadata.get("ticket_id", "?")
        title = h.node.metadata.get("title", "")
        score = f"{h.score:.3f}" if h.score is not None else "  -  "
        print(f"     [{score}] {tid} — {title}")


def build_documents() -> list[Document]:
    docs = []
    for t in load_tickets():
        doc = Document(text=ticket_to_text(t), metadata=ticket_metadata(t), doc_id=t["ticket_id"])
        # Keep metadata out of the embedding text (it's already in ticket_to_text)
        doc.excluded_embed_metadata_keys = list(doc.metadata.keys())
        doc.excluded_llm_metadata_keys = list(doc.metadata.keys())
        docs.append(doc)
    return docs


def main() -> None:
    require_api_key()
    Settings.llm = OpenAI(model=CHAT_MODEL, temperature=0)
    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
    Settings.chunk_size = 1024  # tickets are short — one node per ticket

    docs = build_documents()
    print(f"Loaded {len(docs)} tickets as LlamaIndex Documents")

    # ---- Build indexes -----------------------------------------------------
    banner("Building the five indexes")
    timings = {}

    t0 = time.time()
    vector_index = VectorStoreIndex.from_documents(docs)
    timings["Vector"] = time.time() - t0

    t0 = time.time()
    summary_index = SummaryIndex.from_documents(docs)
    timings["Summary"] = time.time() - t0

    t0 = time.time()
    tree_index = TreeIndex.from_documents(docs, num_children=5)  # LLM summarizes groups of 5
    timings["Tree"] = time.time() - t0

    t0 = time.time()
    # SimpleKeywordTableIndex extracts keywords with regex (no LLM cost).
    # Swap in KeywordTableIndex to have the LLM extract keywords instead.
    keyword_index = SimpleKeywordTableIndex.from_documents(docs)
    timings["Keyword"] = time.time() - t0

    nodes = list(vector_index.docstore.docs.values())
    hybrid = HybridRetriever(vector_index.as_retriever(similarity_top_k=5), nodes, top_k=3)

    for name, secs in timings.items():
        print(f"  {name:<8} index built in {secs:5.2f}s")

    retrievers = {
        "1. Vector Index": vector_index.as_retriever(similarity_top_k=3),
        "2. Summary Index (embedding mode)": summary_index.as_retriever(
            retriever_mode="embedding", similarity_top_k=3
        ),
        "3. Tree Index (select_leaf)": tree_index.as_retriever(retriever_mode="select_leaf"),
        "4. Keyword Table Index": keyword_index.as_retriever(num_chunks_per_query=3),
        "5. Hybrid (BM25 + Vector, RRF)": hybrid,
    }

    # ---- Compare retrieval -------------------------------------------------
    for query in TEST_QUERIES:
        banner(f"Query: {query}")
        for name, retriever in retrievers.items():
            print(f"\n  {name}")
            t0 = time.time()
            try:
                hits = retriever.retrieve(query)[:3]
                print_hits(hits)
            except Exception as exc:  # keep the comparison going if one strategy fails
                print(f"     ⚠️  {type(exc).__name__}: {exc}")
            print(f"     ({time.time() - t0:.2f}s)")

    # ---- Full query engine on the vector index -----------------------------
    banner("Answer generation with the Vector Index query engine")
    engine = vector_index.as_query_engine(similarity_top_k=3)
    question = "How do we fix intermittent 502 errors from the API?"
    response = engine.query(question)
    print(f"Q: {question}\nA: {response}\n")
    print("Sources:")
    for s in response.source_nodes:
        print(f"  - {s.node.metadata['ticket_id']} ({s.score:.3f})")

    # ---- Summary index shines for 'summarize everything' -------------------
    banner("Summary Index: questions about ALL tickets")
    summary_engine = summary_index.as_query_engine(response_mode="tree_summarize")
    question = "Which categories of issues appear most often, and what themes repeat?"
    print(f"Q: {question}\nA: {summary_engine.query(question)}")

    banner("✅ Module 3 complete")
    print("Key takeaway: vector search is the best default; keywords catch exact codes;")
    print("hybrid gives you both. Summary/Tree indexes trade cost for big-picture answers.")


if __name__ == "__main__":
    main()
