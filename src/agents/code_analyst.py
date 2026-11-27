import os
import time
from dataclasses import dataclass
from smolagents import LiteLLMModel, LocalPythonExecutor, ToolCallingAgent, tool

from src.config import get_settings
from src.rag.retriever import HybridRetriever


@dataclass
class ComputationResult:
    """Structured result returned by the smolagents engineering computation sub-agent."""

    summary: str
    steps_taken: int


def _build_python_math_tool():
    """Create a sandboxed Python execution @tool using smolagents' LocalPythonExecutor."""
    python_executor = LocalPythonExecutor(
        additional_authorized_imports=["math", "statistics"]
    )
    python_executor.send_tools({})

    @tool
    def run_python_math(code: str) -> str:
        """Executes Python code inside smolagents' restricted LocalPythonExecutor for exact unit conversions and engineering calculations.

        Args:
            code: Valid Python snippet that computes values and prints the results using print().
        """
        try:
            res = python_executor(code)
            if hasattr(res, "logs") or hasattr(res, "output"):
                logs = getattr(res, "logs", "") or ""
                output = getattr(res, "output", None)
            elif isinstance(res, tuple):
                output, logs = res[0], res[1]
            else:
                output, logs = res, ""

            parts = []
            if logs:
                parts.append(str(logs).strip())
            if output is not None:
                parts.append(f"Result: {output}")
            return "\n".join(parts) if parts else "Executed successfully (no output printed)."
        except Exception as exc:
            return f"Python execution error: {exc}"

    return run_python_math


class EngineeringCodeAnalyst:
    """Hugging Face smolagents ToolCallingAgent + LocalPythonExecutor for verified engineering math."""

    def __init__(self, max_steps: int = 4) -> None:
        settings = get_settings()

        if settings.groq_api_key:
            os.environ["GROQ_API_KEY"] = settings.groq_api_key

        self.model = LiteLLMModel(
            model_id=f"groq/{settings.agent_tool_model}",
            temperature=settings.llm_temperature,
            max_tokens=1024,
            num_retries=2,
        )
        math_tool = _build_python_math_tool()

        self.agent = ToolCallingAgent(
            tools=[math_tool],
            model=self.model,
            max_steps=max_steps,
            verbosity_level=1,
        )

    def compute(self, question: str, evidence_context: str = "") -> ComputationResult:
        """Run verified Python calculations grounded in retrieved datasheet evidence."""
        task = (
            f"Question:\n{question}\n\n"
            f"Retrieved Datasheet Evidence:\n{evidence_context}\n\n"
            "Instructions:\n"
            "1. Extract only the exact numbers needed from the evidence above.\n"
            "2. Call run_python_math ONCE with short, clean Python (under 10 lines, NO comments) to compute average current (mA), battery life in hours, and days (hours / 24).\n"
            "3. Call final_answer with a concise response citing the exact PDF filename and page numbers. Never reply with plain text.\n"
        )

        for attempt in range(1, 4):
            try:
                final_output = str(self.agent.run(task)).strip()
                if final_output.startswith("Calling tools:"):
                    final_output = (
                        "Could not complete automated calculation within step limits. "
                        "Please specify the exact battery capacity (mAh) and active duty cycle (%)."
                    )
                step_count = (
                    len(self.agent.memory.steps)
                    if hasattr(self.agent, "memory")
                    else 0
                )
                return ComputationResult(
                    summary=final_output,
                    steps_taken=step_count,
                )
            except Exception as exc:
                msg = str(exc).lower()
                if ("429" in msg or "rate limit" in msg or "tpm" in msg) and attempt < 3:
                    wait_sec = 20 * attempt
                    print(f"\n[RateLimit] Waiting {wait_sec}s before retry {attempt}/3...")
                    time.sleep(wait_sec)
                    continue
                raise

        raise RuntimeError("EngineeringCodeAnalyst failed after retries.")


if __name__ == "__main__":
    # 1. Retrieve exact datasheet pages using Phase 2 HybridRetriever
    retriever = HybridRetriever(candidate_k=8, final_top_n=1)
    active_chunks = retriever.retrieve("ESP32-S3 Wi-Fi Active RX current mA")
    sleep_chunks = retriever.retrieve("ESP32-S3 Deep-sleep current RTC memory peripherals powered up uA")
    evidence = retriever.format_context(active_chunks + sleep_chunks)

    # 2. Hand verified evidence to Phase 3 smolagents EngineeringCodeAnalyst
    analyst = EngineeringCodeAnalyst(max_steps=4)
    question = (
        "Using the ESP32-S3 Wi-Fi Active RX current (HT20, in mA) and Deep-sleep current "
        "when RTC memory and RTC peripherals are powered up (in uA), calculate exact average "
        "current (mA) and battery life in days on a 2000 mAh battery at a 5% active duty cycle."
    )
    result = analyst.compute(question=question, evidence_context=evidence)
    print("\n================ SMOLAGENTS ANALYST RESULT ================")
    print(result.summary)
    print(f"\nMemory steps recorded: {result.steps_taken}")