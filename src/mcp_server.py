from fastmcp import FastMCP

from src.graph.workflow import SpecTraceOrchestrator
from src.rag.retriever import HybridRetriever

mcp = FastMCP("SpecTrace-Datasheet-MCP")

_retriever: HybridRetriever | None = None
_orchestrator: SpecTraceOrchestrator | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(candidate_k=8, final_top_n=3)
    return _retriever


def _get_orchestrator() -> SpecTraceOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SpecTraceOrchestrator(max_rewrites=1)
    return _orchestrator


@mcp.tool
def search_datasheets(query: str) -> str:
    """Search ESP32-S3 and STM32F401 PDF datasheets using Hybrid RAG (ChromaDB + BM25 + RRF + Cross-Encoder Reranking) and return cited excerpts."""
    retriever = _get_retriever()
    chunks = retriever.retrieve(query)
    if not chunks:
        return "No relevant datasheet excerpts found."
    return retriever.format_context(chunks)


@mcp.tool
def calculate_battery_life(
    active_current_ma: float,
    sleep_current_ua: float,
    active_duty_cycle_percent: float,
    battery_capacity_mah: float,
) -> dict:
    """Deterministically calculate average current (mA) and battery runtime (hours & days) for a duty-cycled microcontroller."""
    duty = active_duty_cycle_percent / 100.0
    sleep_ma = sleep_current_ua / 1000.0
    avg_ma = (duty * active_current_ma) + ((1.0 - duty) * sleep_ma)
    hours = battery_capacity_mah / avg_ma
    days = hours / 24.0
    return {
        "average_current_ma": round(avg_ma, 4),
        "battery_life_hours": round(hours, 2),
        "battery_life_days": round(days, 2),
    }


@mcp.tool
def ask_spectrace(question: str, thread_id: str = "mcp-session") -> dict:
    """Run a full multi-agent SpecTrace query (LangGraph Corrective RAG + smolagents Python math + citation verification)."""
    result = _get_orchestrator().ask(question, thread_id=thread_id)
    return {
        "answer": result["final_answer"],
        "intent": result["intent"],
        "verified": result["verified"],
        "rewrites": result["rewrite_count"],
        "web_fallback": result["used_web_fallback"],
    }


@mcp.resource("datasheets://catalog")
def list_indexed_datasheets() -> str:
    """Return the list of hardware datasheets currently indexed in SpecTrace."""
    return (
        "Indexed Datasheets:\n"
        "- esp32_s3_datasheet.pdf (ESP32-S3-WROOM-1 & WROOM-1U v1.8)\n"
        "- stm32f401_datasheet.pdf (STM32F401xB/C)"
    )


if __name__ == "__main__":
    mcp.run()