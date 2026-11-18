"""
Module 5: Two-Layer RAG Evaluation
==================================

Layer 1 — Retrieval: did we fetch the right tickets?
    Precision@K, Recall@K, F1@K, MRR, Hit Rate
Layer 2 — Generation: is the answer faithful and complete?
    Groundedness and Completeness (LLM-as-judge), ROUGE-L vs a reference

Splitting the layers tells you WHERE a failure happens:
    bad retrieval  → fix chunking / embeddings / k / hybrid search
    good retrieval, bad answer → fix the prompt or the model

Run:
  cd modules/5_evaluation
  python demo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from rouge_score import rouge_scorer

from shared.utils import (
    CHAT_MODEL,
    get_langchain_embeddings,
    banner,
    load_tickets,
    require_api_key,
    ticket_metadata,
    ticket_to_text,
)

HERE = Path(__file__).parent
QUERIES_PATH = HERE / "evaluation_queries.json"
REPORT_CSV = HERE / "evaluation_report.csv"
REPORT_JSON = HERE / "evaluation_report.json"
K_VALUES = [1, 3, 5]
GEN_K = 3

NO_ANSWER = "I don't have enough information in the ticket history to answer that."

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using ONLY the tickets in the context. If the answer is not there, reply exactly: \""
     + NO_ANSWER + "\". Cite ticket IDs like [TICK-001]."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

JUDGE_PROMPT = """You are grading a support assistant's answer.

CONTEXT (the only facts the assistant was allowed to use):
{context}

QUESTION: {question}

REFERENCE ANSWER (what a complete answer should cover): {reference}

ASSISTANT ANSWER: {answer}

Score two things from 1 to 5:
- groundedness: 5 = every claim is supported by the CONTEXT; 1 = mostly unsupported or invented.
  A refusal that makes no claims counts as fully grounded (5).
- completeness: 5 = covers all key points of the REFERENCE; 1 = misses nearly everything.
  If the reference says the history does not cover the topic, a clear refusal scores 5
  and any attempted answer scores 1.

Return JSON only: {{"groundedness": <int>, "completeness": <int>, "reasoning": "<one sentence>"}}"""


# ---------------------------------------------------------------------------
# Layer 1: retrieval metrics
# ---------------------------------------------------------------------------
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(1 for r in top if r in relevant) / k if k else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for r in retrieved[:k] if r in relevant) / len(relevant)


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1 / i
    return 0.0


def evaluate_retrieval(store: FAISS, queries: list[dict]) -> pd.DataFrame:
    rows = []
    max_k = max(K_VALUES)
    for q in queries:
        relevant = set(q["relevant_ticket_ids"])
        if not relevant:
            continue  # retrieval metrics are undefined for unanswerable queries
        retrieved = [d.metadata["ticket_id"] for d in store.similarity_search(q["query"], k=max_k)]
        row = {"query_id": q["query_id"], "query": q["query"], "retrieved": retrieved,
               "relevant": sorted(relevant), "mrr": reciprocal_rank(retrieved, relevant)}
        for k in K_VALUES:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            row[f"p@{k}"], row[f"r@{k}"], row[f"f1@{k}"] = p, r, f1(p, r)
            row[f"hit@{k}"] = 1.0 if any(x in relevant for x in retrieved[:k]) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Layer 2: generation metrics
# ---------------------------------------------------------------------------
def judge(client: OpenAI, context: str, question: str, reference: str, answer: str) -> dict:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            context=context, question=question, reference=reference, answer=answer)}],
    )
    try:
        data = json.loads(response.choices[0].message.content)
        return {
            "groundedness": int(data.get("groundedness", 0)),
            "completeness": int(data.get("completeness", 0)),
            "reasoning": data.get("reasoning", ""),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"groundedness": 0, "completeness": 0, "reasoning": "judge returned invalid JSON"}


def evaluate_generation(store: FAISS, queries: list[dict]) -> pd.DataFrame:
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    chain = RAG_PROMPT | llm | StrOutputParser()
    client = OpenAI()
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rows = []
    for q in queries:
        docs = store.similarity_search(q["query"], k=GEN_K)
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        answer = chain.invoke({"context": context, "question": q["query"]})
        scores = judge(client, context, q["query"], q["reference_answer"], answer)
        rows.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "answer": answer,
            "groundedness": scores["groundedness"] / 5,   # normalize to 0-1
            "completeness": scores["completeness"] / 5,
            "rougeL": rouge.score(q["reference_answer"], answer)["rougeL"].fmeasure,
            "judge_reasoning": scores["reasoning"],
        })
        print(f"  {q['query_id']}: grounded={scores['groundedness']}/5 "
              f"complete={scores['completeness']}/5")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def diagnose(ret_row: pd.Series | None, gen_row: pd.Series) -> str:
    good_retrieval = ret_row is None or ret_row[f"hit@{GEN_K}"] == 1.0
    good_answer = gen_row["groundedness"] >= 0.8 and gen_row["completeness"] >= 0.6
    if good_retrieval and good_answer:
        return "✅ OK"
    if not good_retrieval:
        return "🔍 Retrieval failure"
    return "✍️  Generation failure"


def main() -> None:
    require_api_key()
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    tickets = load_tickets()

    banner("Building FAISS index")
    docs = [Document(page_content=ticket_to_text(t), metadata=ticket_metadata(t)) for t in tickets]
    store = FAISS.from_documents(docs, get_langchain_embeddings())
    print(f"Indexed {len(docs)} tickets | {len(queries)} evaluation queries")

    # ---- Layer 1 -----------------------------------------------------------
    banner("LAYER 1: Retrieval metrics")
    ret = evaluate_retrieval(store, queries)
    summary_cols = [c for c in ret.columns if c.startswith(("p@", "r@", "f1@", "hit@"))] + ["mrr"]
    print("\nPer-query (K=3):")
    print(ret[["query_id", "p@3", "r@3", "f1@3", "mrr"]].round(2).to_string(index=False))
    print("\nAverages:")
    means = ret[summary_cols].mean()
    for k in K_VALUES:
        print(f"  K={k}:  Precision {means[f'p@{k}']:.2f} | Recall {means[f'r@{k}']:.2f} | "
              f"F1 {means[f'f1@{k}']:.2f} | Hit rate {means[f'hit@{k}']:.2f}")
    print(f"  MRR: {means['mrr']:.2f}")
    print("\n👉 Precision falls as K grows (more noise); recall rises (more coverage).")

    # ---- Layer 2 -----------------------------------------------------------
    banner("LAYER 2: Generation metrics (LLM-as-judge)")
    gen = evaluate_generation(store, queries)
    print(f"\n  Groundedness: {gen['groundedness'].mean():.2f}")
    print(f"  Completeness: {gen['completeness'].mean():.2f}")
    print(f"  ROUGE-L     : {gen['rougeL'].mean():.2f}  (lexical overlap — a rough signal only)")

    # ---- Diagnosis ---------------------------------------------------------
    banner("Diagnosis: where did each query fail?")
    ret_by_id = {r["query_id"]: r for _, r in ret.iterrows()}
    gen["diagnosis"] = [diagnose(ret_by_id.get(r["query_id"]), r) for _, r in gen.iterrows()]
    for _, r in gen.iterrows():
        print(f"  {r['query_id']}  {r['diagnosis']:<24} {r['query'][:55]}")

    # ---- Save --------------------------------------------------------------
    merged = gen.merge(ret.drop(columns=["query"]), on="query_id", how="left")
    merged.to_csv(REPORT_CSV, index=False)
    REPORT_JSON.write_text(json.dumps({
        "retrieval_averages": {k: round(float(v), 4) for k, v in means.items()},
        "generation_averages": {
            "groundedness": round(float(gen["groundedness"].mean()), 4),
            "completeness": round(float(gen["completeness"].mean()), 4),
            "rougeL": round(float(gen["rougeL"].mean()), 4),
        },
        "per_query": json.loads(merged.to_json(orient="records")),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {REPORT_CSV.name} and {REPORT_JSON.name}")

    banner("✅ Module 5 complete")
    print("Key takeaway: measure retrieval and generation separately so you know what to fix.")


if __name__ == "__main__":
    main()
