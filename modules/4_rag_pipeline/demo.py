"""
Module 4: The Complete RAG Pipeline (LangChain LCEL)
====================================================

Retrieve → Augment → Generate, with anti-hallucination safeguards:
  - strict grounding prompt ("answer ONLY from context")
  - temperature=0
  - similarity threshold: refuse when nothing relevant is retrieved
  - mandatory ticket-ID citations, returned alongside the answer

Run:
  cd modules/4_rag_pipeline
  python demo.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from shared.utils import (
    CHAT_MODEL,
    get_langchain_embeddings,
    banner,
    load_tickets,
    require_api_key,
    ticket_metadata,
    ticket_to_text,
)

VECTORSTORE_DIR = Path(__file__).parent / "rag_vectorstore"
TOP_K = 3
MIN_RELEVANCE = 0.30  # below this, a ticket is treated as "not relevant"

NO_ANSWER = "I don't have enough information in the ticket history to answer that."

GROUNDED_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a support troubleshooting assistant. Answer using ONLY the tickets in the context.\n"
     "Rules:\n"
     "1. If the context does not contain the answer, reply exactly: \"" + NO_ANSWER + "\"\n"
     "2. Do not use outside knowledge, guess, or invent steps, commands or settings.\n"
     "3. Cite the ticket ID in square brackets after each fact, e.g. [TICK-004].\n"
     "4. Give the fix as short numbered steps when the context has steps.\n"
     "5. If tickets disagree, say so and cite both."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

NAIVE_PROMPT = ChatPromptTemplate.from_template(
    "Here is some context:\n{context}\n\nAnswer the question: {question}"
)


# ---------------------------------------------------------------------------
# Build pieces
# ---------------------------------------------------------------------------
def build_vectorstore(embeddings: OpenAIEmbeddings) -> Chroma:
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
    docs = [Document(page_content=ticket_to_text(t), metadata=ticket_metadata(t)) for t in load_tickets()]
    return Chroma.from_documents(
        docs,
        embeddings,
        collection_name="rag_tickets",
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "(no relevant tickets found)"
    return "\n\n---\n\n".join(d.page_content for d in docs)


def make_retriever(store: Chroma):
    """Retriever that drops low-relevance hits — safeguard #3."""
    return store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": TOP_K, "score_threshold": MIN_RELEVANCE},
    )


def build_rag_chain(retriever, llm):
    """LCEL chain that returns {"answer", "sources"}.

    Short-circuits with NO_ANSWER when nothing passes the threshold,
    so the LLM never gets a chance to improvise.
    """
    answer_chain = (
        {"context": lambda x: format_docs(x["docs"]), "question": lambda x: x["question"]}
        | GROUNDED_PROMPT
        | llm
        | StrOutputParser()
    )

    def answer_or_refuse(x: dict) -> str:
        return answer_chain.invoke(x) if x["docs"] else NO_ANSWER

    return (
        RunnableParallel(docs=retriever, question=RunnablePassthrough())
        | RunnableParallel(
            answer=RunnableLambda(answer_or_refuse),
            sources=lambda x: [
                {"ticket_id": d.metadata["ticket_id"], "title": d.metadata["title"]} for d in x["docs"]
            ],
        )
    )


def build_simple_chain(retriever, llm, prompt):
    """The canonical LCEL pattern: retriever | format | prompt | llm | parser."""
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


# ---------------------------------------------------------------------------
# Demos
# ---------------------------------------------------------------------------
def print_result(question: str, result: dict) -> None:
    print(f"\n❓ {question}")
    print(f"💬 {result['answer']}")
    if result["sources"]:
        print("📎 Sources: " + ", ".join(s["ticket_id"] for s in result["sources"]))
    else:
        print("📎 Sources: none (refused)")


def main() -> None:
    require_api_key()
    embeddings = get_langchain_embeddings()
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)  # safeguard #2

    banner("Building vector store")
    store = build_vectorstore(embeddings)
    retriever = make_retriever(store)
    print(f"Indexed tickets into {VECTORSTORE_DIR.name}/ (k={TOP_K}, threshold={MIN_RELEVANCE})")

    # ---- Peek at retrieval scores -----------------------------------------
    banner("PART 1: Retrieval step on its own")
    q = "Pods keep restarting after we deployed a new version"
    print(f"Query: {q}")
    for doc, score in store.similarity_search_with_relevance_scores(q, k=5):
        flag = "✅" if score >= MIN_RELEVANCE else "❌ below threshold"
        print(f"  [{score:.3f}] {doc.metadata['ticket_id']} — {doc.metadata['title']} {flag}")

    # ---- Full grounded pipeline -------------------------------------------
    banner("PART 2: Grounded RAG answers with citations")
    rag = build_rag_chain(retriever, llm)
    in_scope = [
        "How do I fix intermittent 502 Bad Gateway errors?",
        "Our SSIS package works in Visual Studio but fails in SQL Agent. What should I check?",
        "A quality measure numerator dropped a lot compared to last year. What could cause that?",
    ]
    for question in in_scope:
        print_result(question, rag.invoke(question))

    # ---- Anti-hallucination tests -----------------------------------------
    banner("PART 3: Anti-hallucination — questions the tickets can't answer")
    out_of_scope = [
        "What is the capital of Australia?",
        "How do I configure Kafka consumer group rebalancing?",
        "What was the root cause of the payroll outage last Friday?",
    ]
    for question in out_of_scope:
        print_result(question, rag.invoke(question))

    # ---- Naive vs grounded -------------------------------------------------
    banner("PART 4: Naive prompt vs grounded prompt (same retriever)")
    loose_retriever = store.as_retriever(search_kwargs={"k": TOP_K})  # no threshold
    naive = build_simple_chain(loose_retriever, ChatOpenAI(model=CHAT_MODEL, temperature=0.7), NAIVE_PROMPT)
    grounded = build_simple_chain(loose_retriever, llm, GROUNDED_PROMPT)
    question = "How do I configure Kafka consumer group rebalancing?"
    print(f"❓ {question}")
    print(f"\n🙈 Naive (temp 0.7, no rules):\n{naive.invoke(question)[:600]}")
    print(f"\n🛡️  Grounded (temp 0, strict rules):\n{grounded.invoke(question)}")

    # ---- Streaming ---------------------------------------------------------
    banner("PART 5: Streaming the answer token by token")
    question = "How should we handle SQL Server deadlocks during flash sales?"
    print(f"❓ {question}\n💬 ", end="", flush=True)
    for token in grounded.stream(question):
        print(token, end="", flush=True)
    print()

    banner("✅ Module 4 complete")
    print("Key takeaway: grounding comes from layers — retrieval threshold, strict")
    print("prompt, temperature 0 and citations. No single trick is enough.")


if __name__ == "__main__":
    main()
