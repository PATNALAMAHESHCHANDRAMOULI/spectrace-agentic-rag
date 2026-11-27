import re
import time
from typing import Literal, TypedDict
from ddgs import DDGS
from groq import Groq, RateLimitError
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.code_analyst import EngineeringCodeAnalyst
from src.config import get_settings
from src.rag.retriever import HybridRetriever


class SpecTraceState(TypedDict):
    """Typed state passed across every node in the LangGraph workflow."""

    question: str
    search_query: str
    intent: Literal["rag_only", "rag_compute", "web_lookup"]
    evidence_text: str
    has_sufficient_evidence: bool
    rewrite_count: int
    used_web_fallback: bool
    draft_answer: str
    final_answer: str
    verified: bool


class SpecTraceOrchestrator:
    """LangGraph stateful orchestrator combining Hybrid RAG + Corrective Rewrite + Web Fallback + smolagents Code/Math."""

    def __init__(self, max_rewrites: int = 1) -> None:
        settings = get_settings()
        self.client = Groq(api_key=settings.groq_api_key)
        self.fast_model = settings.fast_llm_model
        self.primary_model = settings.primary_llm_model
        self.temperature = settings.llm_temperature
        self.max_rewrites = max_rewrites

        self.retriever = HybridRetriever(candidate_k=8, final_top_n=2)
        self.analyst = EngineeringCodeAnalyst(max_steps=4)
        # In-memory checkpointer enabling stateful execution per `thread_id` (HF Unit 2.3)
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _chat(self, model: str, system_prompt: str, user_prompt: str) -> str:
        """Helper with 429 rate-limit retry so graph nodes never crash on free-tier bursts."""
        for attempt in range(1, 4):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    reasoning_effort="low",
                )
                return (resp.choices[0].message.content or "").strip()
            except RateLimitError:
                if attempt == 3:
                    raise
                wait_s = 12 * attempt
                print(f"[LangGraph RateLimit] Waiting {wait_s}s before retry {attempt}/3...")
                time.sleep(wait_s)
        return ""

    def _web_search(self, query: str, max_results: int = 3) -> str:
        """Query DuckDuckGo for external/live information not present in local PDFs."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No external web search results found."
            lines = []
            for idx, r in enumerate(results, start=1):
                title = r.get("title", "Web Result")
                href = r.get("href", "")
                body = r.get("body", "")
                lines.append(f"[Web {idx}: {title} ({href})]: {body}")
            return "\n\n".join(lines)
        except Exception as exc:
            return f"Web search unavailable ({exc})."

    # --- Node 1: Intent Router ---
    # --- Node 1: Intent Router ---
    def route_query(self, state: SpecTraceState) -> dict:
        """Route queries to web_lookup, rag_compute (numeric math/duty-cycle), or rag_only (datasheet facts)."""
        q_lower = state["question"].lower()
        web_keywords = ["price", "cost", "buy", "latest news", "release date", "web", "online"]
        explicit_math = ["calculate", "compute", "average current", "duty cycle", "how many days", "how many hours"]
        numeric_context = bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:mah|ah|%|ma|ua|µa|v|hz|mhz)\b", q_lower))

        if any(k in q_lower for k in web_keywords):
            intent: Literal["rag_only", "rag_compute", "web_lookup"] = "web_lookup"
        elif any(k in q_lower for k in explicit_math) and numeric_context:
            intent = "rag_compute"
        elif "battery life" in q_lower and numeric_context:
            intent = "rag_compute"
        else:
            intent = "rag_only"

        print(f"[Node: route_query] intent='{intent}'")
        return {"intent": intent, "search_query": state["question"]}

    # --- Node 2: Hybrid RAG Retrieval (or Direct Web Lookup) ---
    def retrieve_docs(self, state: SpecTraceState) -> dict:
        """Retrieve top reranked datasheet chunks (or run web lookup if routed to web_lookup)."""
        if state["intent"] == "web_lookup":
            print(f"[Node: retrieve_docs -> web] query='{state['search_query']}'")
            web_text = self._web_search(state["search_query"])
            return {"evidence_text": web_text, "used_web_fallback": True}

        print(f"[Node: retrieve_docs -> rag] query='{state['search_query']}'")
        chunks = self.retriever.retrieve(state["search_query"])
        formatted = []
        for c in chunks:
            clean = " ".join(c.text.split())[:1300]
            formatted.append(f"[{c.file_name}, p. {c.page_label}]: {clean}")
        evidence_text = (
            "\n\n".join(formatted) if formatted else "No relevant datasheet chunks found."
        )
        return {"evidence_text": evidence_text}

    # --- Node 3: Corrective RAG Grader + Query Rewrite / Web Fallback ---
    def grade_and_rewrite(self, state: SpecTraceState) -> dict:
        """Grade evidence quality; rewrite query once if weak, or fall back to web search if still weak."""
        if state.get("used_web_fallback"):
            print("[Node: grade_and_rewrite] Using web search evidence.")
            return {"has_sufficient_evidence": True}

        verdict = self._chat(
            model=self.fast_model,
            system_prompt=(
                "You are a strict RAG evidence grader for microcontroller datasheets. "
                "Reply with ONLY 'YES' if the retrieved excerpts contain concrete technical values, "
                "tables, or specifications relevant to the question. Reply 'NO' if they only show "
                "a Table of Contents, revision history, or unrelated info."
            ),
            user_prompt=(
                f"Question: {state['question']}\n\n"
                f"Retrieved Evidence:\n{state['evidence_text'][:2200]}"
            ),
        ).upper()

        sufficient = verdict.startswith("YES")
        rewrites_done = state.get("rewrite_count", 0)

        print(
            f"[Node: grade_and_rewrite] sufficient={sufficient} "
            f"(rewrite_count={rewrites_done}/{self.max_rewrites})"
        )

        # First corrective step: rewrite query with datasheet vocabulary and re-retrieve locally
        if not sufficient and rewrites_done < self.max_rewrites:
            rewritten = self._chat(
                model=self.fast_model,
                system_prompt=(
                    "Rewrite the user query into a concise keyword search query using exact "
                    "microcontroller datasheet terms (e.g., 'Run mode', 'Deep-sleep', 'Standby mode', "
                    "'Current consumption', '3.3 V', 'mA', 'uA'). Output ONLY the rewritten query."
                ),
                user_prompt=f"Original question: {state['question']}",
            )
            cleaned_query = rewritten.strip().strip('"') or state["question"]
            print(f"  -> Rewriting query to: '{cleaned_query}'")
            return {
                "has_sufficient_evidence": False,
                "search_query": cleaned_query,
                "rewrite_count": rewrites_done + 1,
            }

        # Second corrective step (Agentic RAG fallback): if local PDFs still lack evidence, search web
        if not sufficient and rewrites_done >= self.max_rewrites:
            print("  -> Local RAG still insufficient; falling back to live web search...")
            web_text = self._web_search(state["question"])
            return {
                "evidence_text": state["evidence_text"] + "\n\n" + web_text,
                "has_sufficient_evidence": True,
                "used_web_fallback": True,
            }

        return {"has_sufficient_evidence": sufficient}

    def _after_grading(self, state: SpecTraceState) -> Literal["retrieve_docs", "compute_or_answer"]:
        """Loop back to retrieve_docs only when a fresh query rewrite was just produced."""
        if (
            not state.get("has_sufficient_evidence", True)
            and state.get("rewrite_count", 0) <= self.max_rewrites
            and state.get("search_query") != state["question"]
        ):
            return "retrieve_docs"
        return "compute_or_answer"

    # --- Node 4: Delegate to smolagents Analyst OR Grounded Synthesizer ---
    def compute_or_answer(self, state: SpecTraceState) -> dict:
        """Execute verified Python math via smolagents or synthesize a grounded answer."""
        if state["intent"] == "rag_compute":
            print("[Node: compute_or_answer] Delegating to smolagents EngineeringCodeAnalyst...")
            comp = self.analyst.compute(
                question=state["question"],
                evidence_context=state["evidence_text"],
            )
            return {"draft_answer": comp.summary}

        print("[Node: compute_or_answer] Synthesizing grounded answer...")
        answer = self._chat(
            model=self.primary_model,
            system_prompt=(
                "You are SpecTrace, a technical engineering assistant. Answer ONLY using the "
                "provided evidence. Cite exact sources using ASCII brackets like "
                "[esp32_s3_datasheet.pdf, p. 30] or [Web 1]. If a specific figure is missing "
                "or ambiguous in the provided evidence, state that clearly rather than guessing."
            ),
            user_prompt=(
                f"Question: {state['question']}\n\n"
                f"Evidence:\n{state['evidence_text']}"
            ),
        )
        return {"draft_answer": answer}

    # --- Node 5: Citation & Grounding Guardrail ---
    def verify_citations(self, state: SpecTraceState) -> dict:
        """Verify that grounded answers cite sources; append a deterministic source note if missing."""
        draft = state.get("draft_answer", "").strip()
        if state.get("used_web_fallback") or state.get("intent") == "web_lookup":
            has_web_ref = bool(re.search(r"\[Web\s*\d+|https?://", draft, re.IGNORECASE))
            if has_web_ref:
                print("[Node: verify_citations] Web citations verified.")
                return {"final_answer": draft, "verified": True}
            print("[Node: verify_citations] Appending web source context.")
            return {
                "final_answer": draft + "\n\nSource: live web search results.",
                "verified": False,
            }

        has_pdf = bool(re.search(r"\.pdf", draft, re.IGNORECASE))
        has_page = bool(re.search(r"\bp\.?\s*\d+", draft, re.IGNORECASE))
        if has_pdf and has_page:
            print("[Node: verify_citations] Datasheet citations verified (PDF + page reference).")
            return {"final_answer": draft, "verified": True}

        print("[Node: verify_citations] Missing explicit ASCII page citation; appending source tags.")
        found_tags = sorted(
            set(
                re.findall(
                    r"\[[^\]]*\.pdf,\s*p\.\s*[^\]]+\]",
                    state.get("evidence_text", ""),
                )
            )
        )
        suffix = (
            "\n\nSources retrieved: " + ", ".join(found_tags)
            if found_tags
            else "\n\nSource: local datasheet index."
        )
        return {"final_answer": draft + suffix, "verified": False}

    def _build_graph(self):
        """Construct and compile the stateful LangGraph StateGraph with MemorySaver."""
        builder = StateGraph(SpecTraceState)

        builder.add_node("route_query", self.route_query)
        builder.add_node("retrieve_docs", self.retrieve_docs)
        builder.add_node("grade_and_rewrite", self.grade_and_rewrite)
        builder.add_node("compute_or_answer", self.compute_or_answer)
        builder.add_node("verify_citations", self.verify_citations)

        builder.add_edge(START, "route_query")
        builder.add_edge("route_query", "retrieve_docs")
        builder.add_edge("retrieve_docs", "grade_and_rewrite")

        builder.add_conditional_edges(
            "grade_and_rewrite",
            self._after_grading,
            {
                "retrieve_docs": "retrieve_docs",
                "compute_or_answer": "compute_or_answer",
            },
        )

        builder.add_edge("compute_or_answer", "verify_citations")
        builder.add_edge("verify_citations", END)

        return builder.compile(checkpointer=self.checkpointer)

    def ask(self, question: str, thread_id: str = "default-session") -> SpecTraceState:
        """Execute a question through the LangGraph workflow under a stateful thread_id."""
        initial_state: SpecTraceState = {
            "question": question,
            "search_query": question,
            "intent": "rag_only",
            "evidence_text": "",
            "has_sufficient_evidence": False,
            "rewrite_count": 0,
            "used_web_fallback": False,
            "draft_answer": "",
            "final_answer": "",
            "verified": False,
        }
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.invoke(initial_state, config=config)


if __name__ == "__main__":
    orchestrator = SpecTraceOrchestrator(max_rewrites=1)

    q_math = (
        "Calculate the ESP32-S3 average current (mA) and battery life in days on a "
        "2000 mAh battery at a 5% active duty cycle using Wi-Fi RX HT20 current (mA) "
        "and Deep-sleep current with RTC memory and RTC peripherals powered up (uA)."
    )
    print("\n==================== TEST: LANGGRAPH -> SMOLAGENTS MATH ====================")
    res = orchestrator.ask(q_math, thread_id="demo-thread-1")
    print("\nFINAL ANSWER:\n", res["final_answer"])
    print(f"Intent: {res['intent']} | Verified: {res['verified']} | Web fallback: {res['used_web_fallback']}")