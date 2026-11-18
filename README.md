# TicketSearch-RAG

A retrieval-augmented generation (RAG) assistant that answers troubleshooting questions from resolved support-ticket history, built step by step in six modules with **OpenAI** embeddings and chat models.

Each module is a single runnable `demo.py` with the concepts explained in comments and printed output.

| # | Module | What it covers | Key tools |
|---|--------|----------------|-----------|
| 1 | Embeddings | OpenAI embeddings, cosine similarity, semantic search, similarity heatmap | `openai`, NumPy, seaborn |
| 2 | Chunking | Fixed, recursive, semantic, Markdown/HTML-aware splitting; Chroma vector store with metadata filters | LangChain splitters, Chroma |
| 3 | Indexing | Vector, Summary, Tree, Keyword Table indexes, plus **hybrid BM25 + vector** with Reciprocal Rank Fusion | LlamaIndex, rank-bm25 |
| 4 | RAG Pipeline | LCEL pipeline, grounding prompt, relevance threshold, citations, refusal, streaming | LangChain LCEL, Chroma |
| 5 | Evaluation | Retrieval (Precision/Recall/F1@K, MRR, hit rate) + generation (LLM-as-judge groundedness/completeness, ROUGE-L), failure diagnosis | FAISS, OpenAI JSON mode |
| 6 | Agentic RAG | Custom tools, OpenAI function calling, conversation memory, multi-step reasoning, agentic vs direct RAG | LangChain `create_agent`, LangGraph |

---

## Quick start

### 1. Python 3.12
Python 3.12 is the tested version. Some `chromadb` releases break on 3.13+.

```bash
python3.12 --version        # macOS/Linux
py -3.12 --version          # Windows
```

### 2. Clone and create a virtual environment
```bash
git clone https://github.com/<your-username>/TicketSearch-RAG.git
cd TicketSearch-RAG

python3.12 -m venv .venv                 # Windows: py -3.12 -m venv .venv
source .venv/bin/activate                # Windows: .\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add your OpenAI key
```bash
cp .env.example .env                     # Windows: Copy-Item .env.example .env
```
Edit `.env`:
```env
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
```
`.env` is git-ignored. Never commit it.

### 5. Smoke test
```bash
cd modules/1_embeddings
python demo.py
```

---

## Running the modules

Run each demo from its own folder:
```bash
cd modules/4_rag_pipeline
python demo.py
```

Run all six from the repo root:

**macOS/Linux**
```bash
for m in 1_embeddings 2_chunking 3_indexing 4_rag_pipeline 5_evaluation 6_agentic_rag; do
  echo -e "\n=== $m ==="; (cd modules/$m && python demo.py)
done
```

**Windows (PowerShell)**
```powershell
foreach ($m in "1_embeddings","2_chunking","3_indexing","4_rag_pipeline","5_evaluation","6_agentic_rag") {
  Write-Host "`n=== $m ===" -ForegroundColor Cyan
  Push-Location "modules/$m"; python demo.py; Pop-Location
}
```

---

## Project structure

```
TicketSearch-RAG/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example                  # copy to .env and add your key
├── .gitignore
├── data/
│   └── synthetic_tickets.json    # 20 resolved support tickets
├── shared/
│   └── utils.py                  # config, data loading, ticket formatting
└── modules/
    ├── 1_embeddings/demo.py
    ├── 2_chunking/demo.py
    ├── 3_indexing/demo.py
    ├── 4_rag_pipeline/demo.py
    ├── 5_evaluation/
    │   ├── demo.py
    │   └── evaluation_queries.json   # ground-truth queries + reference answers
    └── 6_agentic_rag/
        ├── demo.py
        └── tools.py                  # agent tools
```

Generated at runtime (git-ignored): `chroma_db/`, `rag_vectorstore/`, `agent_vectorstore/`, the Module 1 heatmap PNG and the Module 5 evaluation reports.

---

## Using your own tickets

Replace `data/synthetic_tickets.json` with your export. Each ticket needs:

```json
{
  "ticket_id": "TICK-001",
  "title": "...",
  "description": "...",
  "category": "...",
  "priority": "Low | Medium | High | Critical",
  "status": "Resolved",
  "resolution": "...",
  "created_date": "YYYY-MM-DD",
  "tags": ["..."]
}
```

If your field names differ, change `ticket_to_text()` and `ticket_metadata()` in `shared/utils.py`. Then update `modules/5_evaluation/evaluation_queries.json` with queries and relevant ticket IDs from your own data.

---

## Configuration

| Variable | Default | Options |
|----------|---------|---------|
| `OPENAI_API_KEY` | (required) | |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | `text-embedding-3-large` for higher quality |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | any OpenAI chat model with tool calling |

**Cost:** running all six modules on the sample data costs a few cents with the default models.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY is not set` | Create `.env` in the repo root (not inside a module folder). |
| `chromadb` crashes on import (Pydantic error) | Use Python 3.12 and recreate `.venv`. |
| 429 rate-limit errors | Wait about a minute and re-run, or check your usage limits. |
| `ModuleNotFoundError` | Activate the venv and re-run `pip install -r requirements.txt`. |
| Deprecation warnings for `langchain-community` / `langchain-experimental` | Harmless for now; they come from FAISS and `SemanticChunker`. |

---

## License
MIT
