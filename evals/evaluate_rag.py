import time
from dataclasses import dataclass

from src.graph.workflow import SpecTraceOrchestrator


@dataclass
class EvalCase:
    """Single test case for evaluating SpecTrace end-to-end."""

    question: str
    expected_intent: str
    expected_keyword: str


EVAL_SUITE: list[EvalCase] = [
    EvalCase(
        question="What is the STM32F401 Run mode supply current at 84 MHz and 3.3V with peripherals disabled?",
        expected_intent="rag_only",
        expected_keyword="mA",
    ),
    EvalCase(
        question="What is the ESP32-S3 Deep-sleep current when RTC memory is powered up?",
        expected_intent="rag_only",
        expected_keyword="uA",
    ),
    EvalCase(
        question=(
            "Calculate the ESP32-S3 average current (mA) and battery life in days on a "
            "2000 mAh battery at a 5% active duty cycle using Wi-Fi RX HT20 current (mA) "
            "and Deep-sleep current with RTC memory and RTC peripherals powered up (uA)."
        ),
        expected_intent="rag_compute",
        expected_keyword="days",
    ),
]


def run_evaluation() -> None:
    """Run SpecTrace across a small benchmark suite and print summary metrics."""
    orchestrator = SpecTraceOrchestrator(max_rewrites=1)
    total = len(EVAL_SUITE)
    intent_hits = 0
    citation_hits = 0
    keyword_hits = 0
    latencies: list[float] = []

    print(f"\nRunning SpecTrace Evaluation Suite ({total} test cases)...\n")

    for idx, case in enumerate(EVAL_SUITE, start=1):
        start_t = time.perf_counter()
        result = orchestrator.ask(case.question, thread_id=f"eval-thread-{idx}")
        elapsed = time.perf_counter() - start_t
        latencies.append(elapsed)

        answer = result["final_answer"]
        intent_ok = result["intent"] == case.expected_intent
        verified_ok = bool(result["verified"])
        kw_ok = case.expected_keyword.lower() in answer.lower()

        intent_hits += int(intent_ok)
        citation_hits += int(verified_ok)
        keyword_hits += int(kw_ok)

        print(f"[{idx}/{total}] Intent={result['intent']} (expected={case.expected_intent})")
        print(
            f"      CitationVerified={verified_ok} | Keyword('{case.expected_keyword}')={kw_ok} | "
            f"Rewrites={result['rewrite_count']} | Latency={elapsed:.2f}s"
        )

        # Gentle pause between eval cases to respect Groq free-tier TPM limits
        if idx < total:
            time.sleep(8)

    avg_latency = sum(latencies) / max(len(latencies), 1)
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Intent Routing Accuracy : {intent_hits}/{total} ({100 * intent_hits / total:.1f}%)")
    print(f"Citation Verification   : {citation_hits}/{total} ({100 * citation_hits / total:.1f}%)")
    print(f"Expected Unit/Keyword   : {keyword_hits}/{total} ({100 * keyword_hits / total:.1f}%)")
    print(f"Average End-to-End Time : {avg_latency:.2f}s")


if __name__ == "__main__":
    run_evaluation()