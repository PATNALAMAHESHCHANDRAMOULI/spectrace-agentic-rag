# SpecTrace

Multi-agent RAG service for querying microcontroller datasheets (`ESP32-S3-WROOM-1` and `STM32F401xB/C`), verifying page-level citations, and running duty-cycle and battery-life calculations in a restricted Python sandbox.

Built with Python 3.12, LangGraph, LlamaIndex, `smolagents`, ChromaDB, FastAPI, FastMCP, Docker, and Groq.

## Architecture

1. **Ingestion (`src/rag/ingestion.py`)**: Reads raw PDFs page-by-page using `pypdf` (skipping blank pages and preserving `file_name` and `page_label`), chunks text at 512 tokens with 64-token overlap, embeds locally with `BAAI/bge-small-en-v1.5` into ChromaDB, and writes a `SimpleDocumentStore` JSON for BM25.
2. **Two-stage retrieval (`src/rag/retriever.py`)**: Runs dense vector search (ChromaDB) alongside sparse lexical search (`BM25Retriever`), fuses rankings with Reciprocal Rank Fusion (`k=60`), and reranks top candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
3. **Stateful orchestration (`src/graph/workflow.py`)**: A LangGraph `StateGraph` with `MemorySaver` checkpointing routes queries across five nodes:
   - `route_query`: Classifies queries into `rag_only`, `rag_compute` (when numeric parameters like `mAh` or `%` are present), or `web_lookup`.
   - `retrieve_docs`: Fetches top reranked datasheet excerpts or queries DuckDuckGo (`ddgs`) for external lookups.
   - `grade_and_rewrite`: Grades whether retrieved chunks contain actual specs rather than table-of-contents pages; rewrites once using datasheet terminology or falls back to web search.
   - `compute_or_answer`: Delegates numeric calculations to a `smolagents` `ToolCallingAgent` backed by `LocalPythonExecutor` (restricted to `math` and `statistics`) or synthesizes a cited answer directly.
   - `verify_citations`: Verifies PDF filename and page references before returning.
4. **Interfaces (`src/api.py`, `src/mcp_server.py`, `src/main.py`)**:
   - **FastAPI service**: Async `/ask` endpoint (offloading blocking retrieval/LLM work via `run_in_threadpool`), `/health` readiness check, Swagger UI at `/docs`, and a lightweight browser interface at `/`.
   - **MCP server**: Exposes `search_datasheets`, `calculate_battery_life`, and `ask_spectrace` tools plus a `datasheets://catalog` resource via `FastMCP`.
   - **CLI & evaluation**: Terminal entrypoint with optional Langfuse telemetry (`src/observability.py`) and an end-to-end benchmark (`eval/evaluate_rag.py`).

## Repository layout

```text
spectrace-agentic-rag/
├── data/raw_pdfs/         # Official ESP32-S3 and STM32F401 datasheets (192 pages)
├── eval/
│   └── evaluate_rag.py    # Routing, citation verification, and latency benchmark
├── src/
│   ├── api.py             # FastAPI REST API, Swagger docs, and browser demo UI
│   ├── mcp_server.py      # Model Context Protocol (FastMCP) server
│   ├── main.py            # CLI entrypoint (single-query and interactive modes)
│   ├── config.py          # Pydantic settings loaded from .env
│   ├── observability.py   # Version-safe Langfuse wrapper (no-op if keys unset)
│   ├── core/              # Pure-Python ReAct loop and baseline tools
│   ├── rag/               # pypdf ingestion, ChromaDB, BM25 + RRF + reranker
│   ├── agents/            # smolagents ToolCallingAgent + LocalPythonExecutor
│   └── graph/             # LangGraph Corrective RAG workflow + MemorySaver
├── Dockerfile
├── .dockerignore
├── .env.example
└── requirements.txt
```

## Implementation notes

- **Explicit `pypdf` extraction**: Used `pypdf.PdfReader` directly instead of generic directory loaders to guarantee accurate page numbering and reset the Chroma collection on re-ingestion to prevent duplicate chunks.
- **Hybrid retrieval + RRF**: Part numbers and table labels (`ESP32-S3-WROOM-1`, `Table 6-7`, `84 MHz`) often miss under dense vector search alone. RRF merges vector and BM25 rankings without score-scale normalization.
- **Routing and sandbox guardrails**: Underspecified prompts (e.g., `"ESP32-S3 battery life"` without `mAh` or duty cycle) route to `rag_only` to avoid wasted compute loops, while numeric calculations execute in `LocalPythonExecutor` with `max_tokens` and raw tool-output guards.
- **Known limitation**: `pypdf` flattens dense multi-column electrical tables, which can occasionally shift adjacent rows; future versions can swap in a layout-aware parser (e.g., Docling or `pdfplumber`) to preserve Markdown table structure.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your keys:

```env
GROQ_API_KEY=your_groq_key
HF_TOKEN=optional_hf_token
LANGFUSE_PUBLIC_KEY=optional
LANGFUSE_SECRET_KEY=optional
LANGFUSE_HOST=[https://cloud.langfuse.com](https://cloud.langfuse.com)
```

## Running SpecTrace

Build the local ChromaDB vector store and BM25 docstore (run once):

```powershell
python -m src.rag.ingestion
```

Start the FastAPI server and browser UI:

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 7860
```

- Web UI: `http://localhost:7860`
- Swagger API Docs: `http://localhost:7860/docs`
- Health Check: `http://localhost:7860/health`

Run via CLI:

```powershell
python -m src.main -q "What is the STM32F401 Run mode supply current at 84 MHz and 3.3V with peripherals disabled?"
```

Start the MCP server (stdio):

```powershell
python -m src.mcp_server
```

Run the evaluation suite:

```powershell
python -m eval.evaluate_rag
```

Run with Docker:

```powershell
docker build -t spectrace .
docker run -p 7860:7860 -e GROQ_API_KEY=your_groq_key spectrace
```