import json
import time
from dataclasses import dataclass, field
from typing import Any
from groq import Groq, RateLimitError

from src.config import get_settings
from src.core.tools import TOOL_REGISTRY, get_openai_tool_schemas


SYSTEM_PROMPT = """You are SpecTrace, a precise hardware and technical documentation assistant.
Solve engineering questions step-by-step using ONLY verified values returned by your tools.
Rules:
1. NEVER guess or invent datasheet specs or battery calculations—always call the provided tools.
2. When comparing components, look up specs and run calculations for every component before answering.
3. Cite the exact numbers returned by the tools in your final answer.
"""


@dataclass
class AgentTraceStep:
    """Stores one tool execution inside the agent loop."""

    step_number: int
    tool_name: str
    tool_args: dict[str, Any]
    observation: str


@dataclass
class AgentRunResult:
    """Final structured output from an agent run."""

    final_answer: str
    steps: list[AgentTraceStep] = field(default_factory=list)
    stopped_early: bool = False


class ReActAgent:
    """Zero-framework Tool-Calling ReAct Agent (HF Agents Course Unit 1)."""

    def __init__(self, max_steps: int | None = None) -> None:
        settings = get_settings()
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.primary_llm_model
        self.temperature = settings.llm_temperature
        self.max_steps = max_steps or settings.max_agent_steps
        self.tool_schemas = get_openai_tool_schemas()

    def _call_llm(self, messages: list[dict[str, Any]]) -> Any:
        """Send messages + JSON tool schemas to the LLM with rate-limit backoff."""
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                    temperature=self.temperature,
                    reasoning_effort="low",
                )
                return response.choices[0].message
            except RateLimitError:
                if attempt == 3:
                    raise
                wait_seconds = 5 * attempt
                print(f"[RateLimit] Waiting {wait_seconds}s before retry {attempt}/3...")
                time.sleep(wait_seconds)
        raise RuntimeError("LLM call failed after retries.")

    def _execute_tool(self, name: str, raw_arguments: str) -> tuple[dict[str, Any], str]:
        """Validate tool name & JSON arguments, execute safely, and return an Observation string."""
        tool = TOOL_REGISTRY.get(name)
        if not tool:
            valid = ", ".join(TOOL_REGISTRY.keys())
            return {}, f"Error: Unknown tool '{name}'. Valid tools: {valid}"

        try:
            parsed_args = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            return {}, f"Error: Invalid JSON arguments ({exc})."

        try:
            observation = tool.func(**parsed_args)
            return parsed_args, observation
        except TypeError as exc:
            return parsed_args, f"Error: Wrong arguments for '{name}': {exc}"

    def run(self, user_query: str, verbose: bool = True) -> AgentRunResult:
        """Execute the Thought -> Tool Call -> Observation loop until final answer or max_steps."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        trace: list[AgentTraceStep] = []

        for step in range(1, self.max_steps + 1):
            assistant_msg = self._call_llm(messages)

            # If the LLM did NOT request any tools, it is ready with the Final Answer
            if not assistant_msg.tool_calls:
                return AgentRunResult(
                    final_answer=(assistant_msg.content or "").strip(),
                    steps=trace,
                )

            # Record the assistant's tool call request in conversation history
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_msg.tool_calls
                    ],
                }
            )

            # Execute each requested tool and append its result as a `role: "tool"` message
            for tc in assistant_msg.tool_calls:
                parsed_args, observation = self._execute_tool(
                    tc.function.name, tc.function.arguments
                )

                if verbose:
                    print(f"\n--- Step {step} | Action: {tc.function.name} ---")
                    print(f"Args:        {parsed_args}")
                    print(f"Observation: {observation}")

                trace.append(
                    AgentTraceStep(
                        step_number=step,
                        tool_name=tc.function.name,
                        tool_args=parsed_args,
                        observation=observation,
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.function.name,
                        "content": observation,
                    }
                )

        return AgentRunResult(
            final_answer="Stopped: reached maximum reasoning steps without a final answer.",
            steps=trace,
            stopped_early=True,
        )


if __name__ == "__main__":
    agent = ReActAgent(max_steps=8)
    question = (
        "Compare the battery life of ESP32-S3 and STM32F401 on a 2000 mAh battery "
        "at 5% active duty cycle. Which microcontroller lasts longer, and by how many days?"
    )
    result = agent.run(question)
    print("\n================ FINAL ANSWER ================")
    print(result.final_answer)
    print(f"\nTotal tool calls: {len(result.steps)} | Stopped early: {result.stopped_early}")