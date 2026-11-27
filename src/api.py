import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.graph.workflow import SpecTraceOrchestrator
from src.observability import ObservabilityTracker, record_trace_io

orchestrator: Optional[SpecTraceOrchestrator] = None
tracker: Optional[ObservabilityTracker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize embeddings, reranker, and LangGraph once at server startup."""
    global orchestrator, tracker
    tracker = ObservabilityTracker()
    orchestrator = SpecTraceOrchestrator(max_rewrites=1)
    yield


app = FastAPI(
    title="SpecTrace API",
    description="Multi-Agent Hybrid RAG & Sandboxed Engineering Calculation Service for ESP32-S3 and STM32F401 Datasheets.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Technical datasheet question.")
    thread_id: str = Field(default="web-session-1", min_length=1, max_length=100)


class AskResponse(BaseModel):
    answer: str
    intent: str
    rewrites: int
    web_fallback: bool
    verified: bool
    latency_ms: int


@app.get("/health")
async def health_check() -> dict:
    """Readiness probe for Docker and cloud platforms."""
    return {
        "status": "ok",
        "service": "spectrace",
        "ready": orchestrator is not None,
    }


@app.post("/ask", response_model=AskResponse)
async def ask_spectrace(payload: AskRequest) -> AskResponse:
    """Run a question through the LangGraph + Hybrid RAG + smolagents pipeline."""
    if orchestrator is None or tracker is None:
        raise HTTPException(status_code=503, detail="SpecTrace orchestrator is still initializing.")

    start_t = time.perf_counter()
    try:
        with tracker.trace_request(
            name="spectrace_api_ask",
            user_id=payload.thread_id,
            metadata={"question": payload.question},
        ) as span:
            # Offload blocking retrieval & LLM calls to a worker thread so the async event loop stays responsive
            result = await run_in_threadpool(
                orchestrator.ask,
                payload.question,
                payload.thread_id,
            )
            elapsed_ms = int((time.perf_counter() - start_t) * 1000)
            record_trace_io(
                span,
                input_data=payload.question,
                output_data=result["final_answer"],
                metadata={
                    "intent": result["intent"],
                    "verified": str(result["verified"]),
                    "rewrites": str(result["rewrite_count"]),
                    "web_fallback": str(result["used_web_fallback"]),
                    "latency_ms": str(elapsed_ms),
                },
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc

    return AskResponse(
        answer=result["final_answer"],
        intent=result["intent"],
        rewrites=result["rewrite_count"],
        web_fallback=result["used_web_fallback"],
        verified=result["verified"],
        latency_ms=elapsed_ms,
    )


DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SpecTrace — Multi-Agent Datasheet Intelligence</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 820px; margin: 40px auto; padding: 0 20px; color: #111827; background: #f9fafb; }
    h1 { margin-bottom: 4px; font-size: 1.75rem; }
    .sub { color: #4b5563; margin-bottom: 24px; font-size: 0.95rem; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    textarea { width: 100%; box-sizing: border-box; padding: 12px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 0.95rem; resize: vertical; min-height: 84px; }
    .samples { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 16px; }
    .chip { background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe; border-radius: 999px; padding: 6px 12px; font-size: 0.82rem; cursor: pointer; }
    .chip:hover { background: #e0e7ff; }
    button.primary { background: #111827; color: #fff; border: none; border-radius: 8px; padding: 10px 20px; font-size: 0.95rem; cursor: pointer; }
    button.primary:disabled { opacity: 0.6; cursor: not-allowed; }
    .result { margin-top: 20px; display: none; }
    .answer { white-space: pre-wrap; line-height: 1.55; font-size: 0.96rem; margin-top: 10px; }
    .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
    .badge { font-size: 0.78rem; padding: 4px 10px; border-radius: 6px; background: #f3f4f6; color: #374151; border: 1px solid #e5e7eb; }
    .footer { margin-top: 18px; font-size: 0.85rem; color: #6b7280; }
    a { color: #2563eb; text-decoration: none; }
  </style>
</head>
<body>
  <h1>SpecTrace</h1>
  <div class="sub">Multi-Agent Hybrid RAG + Sandboxed Python Math for ESP32-S3 &amp; STM32F401 Datasheets</div>

  <div class="card">
    <textarea id="q" placeholder="Ask a datasheet question (e.g., STM32F401 Run mode current or ESP32-S3 battery life calculation)..."></textarea>
    <div class="samples">
      <button class="chip" onclick="setQ('What is the STM32F401 Run mode supply current at 84 MHz and 3.3V with peripherals disabled?')">STM32F401 Run current (RAG)</button>
      <button class="chip" onclick="setQ('Calculate the ESP32-S3 average current (mA) and battery life in days on a 2000 mAh battery at a 5% active duty cycle using Wi-Fi RX HT20 current (mA) and Deep-sleep current with RTC memory and RTC peripherals powered up (uA).')">ESP32-S3 Battery Life (RAG + Python Math)</button>
    </div>
    <button id="btn" class="primary" onclick="runQuery()">Ask SpecTrace</button>

    <div id="result" class="result">
      <div id="badges" class="badges"></div>
      <div id="answer" class="answer"></div>
    </div>
  </div>

  <div class="footer">
    API Docs: <a href="/docs" target="_blank">/docs (Swagger UI)</a> &middot;
    Health: <a href="/health" target="_blank">/health</a>
  </div>

  <script>
    function setQ(text) { document.getElementById('q').value = text; }
    async function runQuery() {
      const q = document.getElementById('q').value.trim();
      if (!q) return;
      const btn = document.getElementById('btn');
      const resBox = document.getElementById('result');
      const ans = document.getElementById('answer');
      const badges = document.getElementById('badges');

      btn.disabled = true;
      btn.textContent = 'Running pipeline...';
      resBox.style.display = 'block';
      badges.innerHTML = '';
      ans.textContent = 'Retrieving datasheet chunks, grading evidence, and verifying citations...';

      try {
        const resp = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q, thread_id: 'web-ui' })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || 'Request failed');

        badges.innerHTML = [
          `intent: ${data.intent}`,
          `verified: ${data.verified}`,
          `rewrites: ${data.rewrites}`,
          `web_fallback: ${data.web_fallback}`,
          `latency: ${data.latency_ms} ms`
        ].map(t => `<span class="badge">${t}</span>`).join('');
        ans.textContent = data.answer;
      } catch (err) {
        ans.textContent = 'Error: ' + err.message;
      } finally {
        btn.disabled = false;
        btn.textContent = 'Ask SpecTrace';
      }
    }
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    """Serve lightweight browser interface for recruiters and live demos."""
    return DEMO_HTML