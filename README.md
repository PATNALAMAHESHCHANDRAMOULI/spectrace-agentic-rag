# SpecTrace

Multi-agent RAG pipeline for querying hardware datasheets (`ESP32-S3-WROOM-1` and `STM32F401xB/C`), verifying page-level citations, and running duty-cycle/battery-life calculations in a restricted Python sandbox.

Built with Python 3.12, LangGraph, LlamaIndex, `smolagents`, ChromaDB, and Groq.

## How it works

1. **Ingestion (`src/rag/ingestion.py`)**: Extracts text page-by-page from raw PDFs using `pypdf` (skipping blank pages and keeping `file_name` + `page_label` metadata), splits into 512-token chunks with 64-token overlap, embeds locally with `BAAI/bge-small-en-v1.5` into ChromaDB, and persists a `SimpleDocumentStore` JSON for BM25.
2. **Two-stage retrieval (`src/rag/retriever.py`)**: Runs dense vector search (ChromaDB) and sparse lexical search (`BM25Retriever`), merges both rankings with Reciprocal Rank Fusion (`k=60`), and reranks candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
3. **Stateful orchestration (`src/graph/workflow.py`)**: A LangGraph `StateGraph` with `MemorySaver` checkpointing routes queries across five nodes:
   - `route_query`: Routes to `rag_only`, `rag_compute`, or `web_lookup`.
   - `retrieve_docs`: Fetches top reranked datasheet chunks (or queries DuckDuckGo via `ddgs` for external lookups).
   - `grade_and_rewrite`: Grades whether retrieved excerpts contain actual specs rather than table-of-contents/revision pages. If weak, rewrites the query once using datasheet terminology and re-retrieves; falls back to web search if local PDFs still don't cover it.
   - `compute_or_answer`: Delegates math/comparison tasks to `smolagents` (`ToolCallingAgent` + `LocalPythonExecutor` restricted to `math` and `statistics`) or synthesizes a grounded answer directly.
   - `verify_citations`: Checks that the output includes PDF filename and page references before returning.
4. **CLI & evaluation (`src/main.py`, `eval/evaluate_rag.py`)**: Single-query and interactive terminal entrypoint with optional Langfuse tracing (`src/observability.py`) and an evaluation script tracking intent routing, citation verification, expected units, and latency.

## Repository layout

```text
spectrace-agentic-rag/
├── data/raw_pdfs/         # Official ESP32-S3 and STM32F401 datasheets (192 pages total)
├── eval/
│   └── evaluate_rag.py    # End-to-end routing, citation, and latency benchmark
├── src/
│   ├── config.py          # Pydantic settings and model IDs loaded from .env
│   ├── main.py            # CLI entrypoint
│   ├── observability.py   # Version-safe Langfuse wrapper (no-op if keys unset)
│   ├── core/              # Pure-Python ReAct loop and baseline tools
│   ├── rag/               # pypdf ingestion, ChromaDB, BM25 + RRF + reranker
│   ├── agents/            # smolagents ToolCallingAgent + LocalPythonExecutor
│   └── graph/             # LangGraph Corrective RAG workflow + MemorySaver
├── .env.example
└── requirements.txt
```

## Notes on implementation choices

- **Explicit `pypdf` extraction**: Used `pypdf.PdfReader` directly instead of generic directory loaders to guarantee accurate page numbering and avoid silent binary-read fallbacks. Re-running ingestion resets the Chroma collection to prevent duplicate chunks.
- **Hybrid retrieval + RRF**: Technical part numbers and table labels (`ESP32-S3-WROOM-1`, `Table 6-7`, `84 MHz`) often miss under dense vector search alone. RRF combines vector and BM25 rankings without score-scale normalization issues.
- **Sandboxed math**: Numerical conversions (`uA` to `mA`, duty-cycle averages, battery life) are executed via `LocalPythonExecutor` with only `math` and `statistics` authorized instead of relying on LLM mental math.
- **Model configuration**: Model IDs and RAG parameters live in `src/config.py` so swapping deprecated or rate-limited models requires a one-line change.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your keys:

```env
GROQ_API_KEY=your_groq_key
HF_TOKEN=optional_hf_token
LANGFUSE_PUBLIC_KEY=optional
LANGFUSE_SECRET_KEY=optional
LANGFUSE_HOST=[https://cloud.langfuse.com](https://cloud.langfuse.com)
```

## Usage

Build the local vector index and BM25 docstore (run once after adding PDFs to `data/raw_pdfs/`):

```powershell
python -m src.rag.ingestion
```

Ask a single question:

```powershell
python -m src.main -q "What is the STM32F401 Run mode supply current at 84 MHz and 3.3V with peripherals disabled?"
```

Start interactive mode:

```powershell
python -m src.main
```

Run the evaluation suite:

```powershell
python -m eval.evaluate_rag
```