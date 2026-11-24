from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    """Represents a single capability exposed to an LLM agent via JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., str]

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert this tool into the standard OpenAI/Groq Function Calling JSON Schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# Temporary in-memory spec table for Unit 1 testing.
# In Phase 2, we replace this with real LlamaIndex Hybrid RAG over PDF datasheets.
_MOCK_SPEC_DB: dict[str, dict[str, str | float]] = {
    "ESP32-S3": {
        "architecture": "Dual-core Xtensa LX7 (240 MHz)",
        "operating_voltage": "3.0V - 3.6V",
        "active_current_ma": 95.0,
        "deep_sleep_current_ua": 8.0,
        "notes": "Wi-Fi + Bluetooth 5 LE; deep sleep with RTC memory enabled.",
    },
    "STM32F401": {
        "architecture": "ARM Cortex-M4 (84 MHz)",
        "operating_voltage": "1.7V - 3.6V",
        "active_current_ma": 12.8,
        "deep_sleep_current_ua": 2.4,
        "notes": "Low-power MCU; Stop mode current with RTC off.",
    },
}


def lookup_component_spec(part_number: str) -> str:
    """Look up electrical specs for a microcontroller part number."""
    normalized = part_number.strip().upper()
    spec = _MOCK_SPEC_DB.get(normalized)

    if not spec:
        available = ", ".join(_MOCK_SPEC_DB.keys())
        return f"Error: Part '{part_number}' not found. Available parts: {available}"

    return (
        f"Part: {normalized} | Architecture: {spec['architecture']} | "
        f"Voltage: {spec['operating_voltage']} | "
        f"Active Current: {spec['active_current_ma']} mA | "
        f"Deep Sleep Current: {spec['deep_sleep_current_ua']} uA | "
        f"Notes: {spec['notes']}"
    )


def calculate_battery_life(
    battery_mah: float,
    active_ma: float,
    sleep_ua: float,
    active_duty_percent: float,
) -> str:
    """Calculate exact average current and battery life with uA-to-mA conversion."""
    if battery_mah <= 0:
        return "Error: battery_mah must be greater than 0."
    if not (0.0 <= active_duty_percent <= 100.0):
        return "Error: active_duty_percent must be between 0 and 100."

    duty_fraction = active_duty_percent / 100.0
    sleep_ma = sleep_ua / 1000.0  # Convert microamps (uA) to milliamps (mA)
    avg_current_ma = (active_ma * duty_fraction) + (sleep_ma * (1.0 - duty_fraction))

    if avg_current_ma <= 0:
        return "Error: calculated average current must be greater than 0."

    hours = battery_mah / avg_current_ma
    days = hours / 24.0

    return (
        f"Average Current: {avg_current_ma:.4f} mA | "
        f"Estimated Battery Life: {hours:.2f} hours ({days:.2f} days)"
    )


TOOL_REGISTRY: dict[str, Tool] = {
    "lookup_component_spec": Tool(
        name="lookup_component_spec",
        description="Retrieves datasheet specs (active current in mA, deep sleep current in uA, voltage) for a microcontroller.",
        parameters={
            "type": "object",
            "properties": {
                "part_number": {
                    "type": "string",
                    "description": "Exact microcontroller part number, e.g. 'ESP32-S3' or 'STM32F401'.",
                }
            },
            "required": ["part_number"],
        },
        func=lookup_component_spec,
    ),
    "calculate_battery_life": Tool(
        name="calculate_battery_life",
        description="Computes exact average current and battery runtime in hours and days.",
        parameters={
            "type": "object",
            "properties": {
                "battery_mah": {"type": "number", "description": "Battery capacity in mAh."},
                "active_ma": {"type": "number", "description": "Active mode current in mA."},
                "sleep_ua": {"type": "number", "description": "Deep sleep current in uA."},
                "active_duty_percent": {
                    "type": "number",
                    "description": "Percentage of time spent active (0 to 100).",
                },
            },
            "required": ["battery_mah", "active_ma", "sleep_ua", "active_duty_percent"],
        },
        func=calculate_battery_life,
    ),
}


def get_openai_tool_schemas() -> list[dict[str, Any]]:
    """Return all registered tools formatted for the chat completions `tools` API parameter."""
    return [tool.to_openai_schema() for tool in TOOL_REGISTRY.values()]