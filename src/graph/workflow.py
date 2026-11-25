import re
import time
from typing import Literal, TypedDict
from groq import Groq, RateLimitError
from langgraph.graph import END, START, StateGraph

from src.agents.code_analyst import EngineeringCodeAnalyst
from src.config import get_settings
from src.rag.retriever import HybridRetriever


class SpecTraceState(TypedDict):
    """Typed state passed across every node in the LangGraph workflow."""

    question: str
    search_query: str
    intent: Literal["rag_only", "rag_compute"]
    evidence_text: str
    has_sufficient_evidence: bool
    rewrite_count: int
    draft_answer: str
    final_answer: str
    verified: bool


class SpecTraceOrchestrator:
    """LangGraph stateful orchestrator combining Hybrid RAG + Corrective Rewrite + smolagents Code/Math."""

    def __init__(self, max_rewrites: int = 1) -> None:
        settings = get_settings()
        self.client = Groq(api_key=settings.groq_api_key)
        self.fast_model = settings.fast_llm_model
        self.primary_model = settings.primary_llm_model
        self.temperature = settings.llm_temperature
        self.max_rewrites = max_rewrites

        # Reuse our Phase 2 HybridRetriever and Phase 3 smolagents analyst
        self.retriever = HybridRetriever(candidate_k=8, final_top_n=2)
        self.analyst = EngineeringCodeAnalyst(max_steps=4)
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

    # --- Node 1: Intent Router ---
    def route_query(self, state: SpecTraceState) -> dict:
        """Classify whether the query needs pure datasheet lookup or math/duty-cycle computation."""
        q_lower = state["question"].lower()
        math_keywords = [
            "calculate",
            "battery life",
            "duty cycle",
            "average current",
            "mah",
            "difference",
            "compare",
            "how many days",
            "how many hours",
        ]
        intent: Literal["rag_only", "rag_compute"] = (
            "rag_compute" if any(k in q_lower for k in math_keywords) else "rag_only"
        )
        print(f"[Node: route_query] intent='{intent}'")
        return {"intent": intent, "search_query": state["question"]}

    # --- Node 2: Hybrid RAG Retrieval ---
    def retrieve_docs(self, state: SpecTraceState) -> dict:
        """Retrieve top reranked datasheet chunks with exact filename and page metadata."""
        print(f"[Node: retrieve_docs] query='{state['search_query']}'")
        chunks = self.retriever.retrieve(state["search_query"])
        formatted = []
        for c in chunks:
            clean = " ".join(c.text.split())[:1300]
            formatted.append(f"[{c.file_name}, p. {c.page_label}]: {clean}")
        evidence_text = (
            "\n\n".join(formatted) if formatted else "No relevant datasheet chunks found."
        )
        return {"evidence_text": evidence_text}

    # --- Node 3: Corrective RAG Relevance Grader + Query Rewriter ---
    def grade_and_rewrite(self, state: SpecTraceState) -> dict:
        """Grade whether retrieved chunks contain concrete specs; if not, rewrite query once."""
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

        # If evidence is weak and we still have rewrite budget, rewrite query using datasheet terms
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

        return {"has_sufficient_evidence": sufficient}

    # --- Conditional Edge after grading ---
    def _after_grading(self, state: SpecTraceState) -> Literal["retrieve_docs", "compute_or_answer"]:
        """Loop back to retrieve_docs if query was rewritten; otherwise proceed to answer/compute."""
        if (
            not state.get("has_sufficient_evidence", True)
            and state.get("rewrite_count", 0) <= self.max_rewrites
            and state.get("search_query") != state["question"]
        ):
            # Only loop when a rewrite actually happened on this turn
            return "retrieve_docs"
        return "compute_or_answer"

    # --- Node 4: Delegate to smolagents Code/Math Analyst OR Grounded RAG Synthesizer ---
    def compute_or_answer(self, state: SpecTraceState) -> dict:
        """Execute verified Python math via smolagents or synthesize a grounded datasheet answer."""
        if state["intent"] == "rag_compute":
            print("[Node: compute_or_answer] Delegating to smolagents EngineeringCodeAnalyst...")
            comp = self.analyst.compute(
                question=state["question"],
                evidence_context=state["evidence_text"],
            )
            return {"draft_answer": comp.summary}

        print("[Node: compute_or_answer] Synthesizing grounded datasheet answer...")
        answer = self._chat(
            model=self.primary_model,
            system_prompt=(
                "You are SpecTrace, a technical datasheet assistant. Answer ONLY using the "
                "provided datasheet excerpts. Every factual number or spec MUST include its exact "
                "citation like [esp32_s3_datasheet.pdf, p. 30]. If a value is missing from the "
                "excerpts, explicitly state that it is not found in the retrieved excerpts."
            ),
            user_prompt=(
                f"Question: {state['question']}\n\n"
                f"Evidence:\n{state['evidence_text']}"
            ),
        )
        return {"draft_answer": answer}

    # --- Node 5: Citation & Grounding Guardrail ---
    def verify_citations(self, state: SpecTraceState) -> dict:
        """Verify that the answer includes PDF/page citations; append a grounding note if missing."""
        draft = state.get("draft_answer", "").strip()
        has_pdf_citation = bool(re.search(r"\.pdf", draft, re.IGNORECASE))
        has_page_ref = bool(re.search(r"\bp\.?\s*\d+", draft, re.IGNORECASE))

        if has_pdf_citation and has_page_ref:
            print("[Node: verify_citations] Citations verified (PDF + page reference present).")
            return {"final_answer": draft, "verified": True}

        print("[Node: verify_citations] Missing explicit page citation; appending source note.")
        # Extract unique [filename, p. X] tags present in retrieved evidence
        found_tags = sorted(
            set(
                re.findall(
                    r"\[[^\]]*\.pdf,\s*p\.\s*[^\]]+\]",
                    state.get("evidence_text", ""),
                )
            )
        )
        source_line = (
            "\n\nSources retrieved: " + ", ".join(found_tags)
            if found_tags
            else "\n\nSource: local datasheet index (see retrieved excerpts)."
        )
        return {"final_answer": draft + source_line, "verified": False}

    def _build_graph(self):
        """Construct and compile the stateful LangGraph StateGraph."""
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

        return builder.compile()

    def ask(self, question: str) -> SpecTraceState:
        """Run a question through the full LangGraph workflow and return the final state."""
        initial_state: SpecTraceState = {
            "question": question,
            "search_query": question,
            "intent": "rag_only",
            "evidence_text": "",
            "has_sufficient_evidence": False,
            "rewrite_count": 0,
            "draft_answer": "",
            "final_answer": "",
            "verified": False,
        }
        return self.graph.invoke(initial_state)


if __name__ == "__main__":
    orchestrator = SpecTraceOrchestrator(max_rewrites=1)

    q1 = (
        "What is the ESP32-S3 Deep-sleep current when RTC memory is powered up and "
        "RTC peripherals are powered down?"
    )
    print("\n==================== TEST 1: GROUNDED RAG LOOKUP ====================")
    res1 = orchestrator.ask(q1)
    print("\nFINAL ANSWER:\n", res1["final_answer"])
    print(f"Verified citations: {res1['verified']} | Rewrites: {res1['rewrite_count']}")