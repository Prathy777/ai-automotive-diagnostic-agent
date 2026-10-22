"""Optional one-call live check for Google-hosted Gemma (not part of unit tests)."""
import asyncio
import json
import logging
import os
import time

from .providers.gemma import GemmaProvider
from .providers.base import ProviderRequestError
from .tools.registry import tool_descriptions


async def main() -> None:
    if not os.getenv("GEMMA_API_KEY"):
        raise SystemExit("Set GEMMA_API_KEY before running this optional smoke test.")
    diagnostics = os.getenv("GEMMA_DIAGNOSTICS", "0").casefold() in {"1", "true", "yes", "on"}
    if diagnostics:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    provider = GemmaProvider()
    if diagnostics:
        logging.getLogger(__name__).info(
            "gemma_smoke_start endpoint=%s model=%s started_at=%s timeout_seconds=%.1f",
            provider.endpoint, provider.model_name, time.strftime("%Y-%m-%dT%H:%M:%S%z"), provider.timeout,
        )
    state = {
        "initial_telemetry": {"rpm": 812, "engine_temperature": 91.4, "engine_vibration": 0.78,
                              "fuel_consumption": 3.1, "battery_voltage": 13.8},
        "observations": [{"id": "obs_001", "kind": "initial_telemetry", "tool": "observe_vehicle",
                          "data": {"rpm": 812, "engine_temperature": 91.4, "engine_vibration": 0.78,
                                   "fuel_consumption": 3.1, "battery_voltage": 13.8}}],
        "hypotheses": [], "previous_actions": [], "step": 1, "remaining_action_budget": 8,
    }
    try:
        decision = await provider.decide(state, tool_descriptions())
    except ProviderRequestError as exc:
        raise SystemExit(f"Gemma smoke test failed: {exc}") from None
    print(json.dumps(decision.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
