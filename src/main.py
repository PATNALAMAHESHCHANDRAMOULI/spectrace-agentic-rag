import argparse
from src.graph.workflow import SpecTraceOrchestrator
from src.observability import ObservabilityTracker, record_trace_io


def run_cli(question: str | None = None, thread_id: str = "cli-session-1") -> None:
    """Run SpecTrace in single-question mode or interactive terminal mode."""
    tracker = ObservabilityTracker()
    orchestrator = SpecTraceOrchestrator(max_rewrites=1)

    if question:
        with tracker.trace_request(
            name="spectrace_ask",
            user_id=thread_id,
            metadata={"question": question},
        ) as trace_span:
            result = orchestrator.ask(question, thread_id=thread_id)
            record_trace_io(
                trace_span,
                input_data=question,
                output_data=result["final_answer"],
                metadata={
                    "intent": result["intent"],
                    "verified": str(result["verified"]),
                    "rewrites": str(result["rewrite_count"]),
                    "web_fallback": str(result["used_web_fallback"]),
                },
            )

        print("\n==================== SPECTRACE ANSWER ====================")
        print(result["final_answer"])
        print(
            f"\n[Telemetry] intent={result['intent']} | "
            f"rewrites={result['rewrite_count']} | "
            f"web_fallback={result['used_web_fallback']} | "
            f"verified={result['verified']}"
        )
        return

    print("SpecTrace interactive mode (type 'exit' or 'quit' to stop).")
    while True:
        user_q = input("\nAsk a datasheet question> ").strip()
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            break
        res = orchestrator.ask(user_q, thread_id=thread_id)
        print("\n--- Answer ---")
        print(res["final_answer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpecTrace Multi-Agent Datasheet Intelligence CLI")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        default=None,
        help="Single technical question to answer; omit for interactive mode.",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default="cli-session-1",
        help="LangGraph checkpoint thread ID for stateful runs.",
    )
    args = parser.parse_args()
    run_cli(question=args.question, thread_id=args.thread_id)