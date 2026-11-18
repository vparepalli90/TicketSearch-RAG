"""
Module 6: Agentic RAG
=====================

Instead of always "retrieve then answer", an agent lets the LLM decide
WHICH tool to call, WHEN, and HOW MANY times — using OpenAI function calling.

What you'll learn:
  1. Custom tools (see tools.py) — semantic search, lookup by ID,
     category filter, statistics
  2. Building an agent with LangChain's create_agent (LangGraph under the hood)
  3. Conversation memory with a checkpointer (thread_id = one conversation)
  4. Multi-step reasoning: the agent chains several tool calls
  5. Agentic vs direct RAG: quality, latency and number of LLM calls

Run:
  cd modules/6_agentic_rag
  python demo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from shared.utils import CHAT_MODEL, banner, require_api_key
from tools import ALL_TOOLS, search_similar_tickets

SYSTEM_PROMPT = """You are a support troubleshooting agent with access to a ticket history.

How to work:
- Use tools to look things up; never answer troubleshooting questions from memory.
- For symptoms, use search_similar_tickets. For ticket IDs, use get_ticket_by_id.
  For "all X tickets" use search_by_category. For counts/trends use get_ticket_statistics.
- You may call several tools, one after another, before answering.
- If the tools return nothing relevant, say the ticket history doesn't cover it.
- Cite ticket IDs in square brackets, e.g. [TICK-004]. Keep answers concise."""


def build_agent():
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    return create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),  # gives the agent memory per thread_id
    )


def run_agent(agent, question: str, thread_id: str, show_trace: bool = True) -> dict:
    """Invoke the agent and print the tool calls it made."""
    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.time()
    before = len(agent.get_state(config).values.get("messages", [])) if show_trace else 0
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config)
    elapsed = time.time() - t0

    new_messages = result["messages"][before:]
    llm_calls = sum(1 for m in new_messages if isinstance(m, AIMessage))
    tool_calls = [tc for m in new_messages if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]

    print(f"\n❓ {question}")
    if show_trace:
        for tc in tool_calls:
            args = ", ".join(f"{k}={v!r}" for k, v in tc["args"].items())
            print(f"   🔧 {tc['name']}({args})")
        for m in new_messages:
            if isinstance(m, ToolMessage):
                print(f"      ↳ {m.content[:90].replace(chr(10), ' ')}...")
    answer = result["messages"][-1].content
    print(f"💬 {answer}")
    print(f"   ⏱  {elapsed:.1f}s | {llm_calls} LLM calls | {len(tool_calls)} tool calls")
    return {"answer": answer, "seconds": elapsed, "llm_calls": llm_calls, "tool_calls": len(tool_calls)}


def direct_rag(question: str) -> dict:
    """Module 4-style fixed pipeline: always retrieve top-3, then answer once."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the tickets below. If they don't cover it, say so. "
                   "Cite ticket IDs like [TICK-001]."),
        ("human", "Tickets:\n{context}\n\nQuestion: {question}"),
    ])
    t0 = time.time()
    context = search_similar_tickets.invoke({"query": question, "k": 3})
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"answer": answer, "seconds": time.time() - t0, "llm_calls": 1}


def main() -> None:
    require_api_key()
    agent = build_agent()

    banner("PART 1: The agent picks the right tool")
    run_agent(agent, "Users are stuck in a login redirect loop. How do we fix it?", "t1")
    run_agent(agent, "Show me the details of TICK-016.", "t2")
    run_agent(agent, "How many tickets do we have, and which category is most common?", "t3")

    banner("PART 2: Multi-step reasoning (several tool calls)")
    run_agent(agent,
              "List the Critical Database tickets, then tell me which fix pattern they "
              "have in common and whether any Infrastructure ticket had a similar root cause.",
              "t4")

    banner("PART 3: Conversation memory (same thread)")
    run_agent(agent, "What did we do about the BigQuery cost spike?", "t5")
    run_agent(agent, "Would the same approach help with the nightly ETL timeout?", "t5")
    run_agent(agent, "Which of those two tickets was opened first?", "t5")

    banner("PART 4: Out-of-scope question")
    run_agent(agent, "How do I set up a Kafka cluster on Kubernetes?", "t6")

    banner("PART 5: Agentic vs Direct RAG")
    comparisons = [
        "How do I fix intermittent 502 errors?",                          # simple — direct is enough
        "How many High priority Reporting tickets are there and what were they about?",  # needs filter + count
    ]
    for q in comparisons:
        d = direct_rag(q)
        print(f"\n❓ {q}")
        print(f"   DIRECT  ({d['seconds']:.1f}s, {d['llm_calls']} LLM call): {d['answer'][:250]}")
        a = run_agent(agent, q, f"cmp-{hash(q)}", show_trace=False)
        print(f"   AGENTIC ({a['seconds']:.1f}s, {a['llm_calls']} LLM calls) — see answer above")

    print("""
When to use which:
  Direct RAG  → simple Q&A, low latency, predictable cost
  Agentic RAG → filters, counts, multi-step questions, follow-ups, multiple sources""")

    banner("✅ Module 6 complete")
    print("Key takeaway: agents decide WHEN and HOW to retrieve — more flexible,")
    print("but slower and costlier than a fixed pipeline.")


if __name__ == "__main__":
    main()
