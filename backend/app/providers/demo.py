"""Explicit deterministic demo decisions. This is not an LLM or Gemma output."""
from typing import Any

from ..agent_schemas import AgentDecision

_TESTS = {
    "misfire": ("run_2000_rpm_test", "Engine misfire", "Engine-correlated vibration and RPM instability support a misfire."),
    "overheating": ("run_cooling_system_test", "Cooling system fault", "High coolant temperature and weak fan response support a cooling fault."),
    "alternator_failure": ("run_electrical_load_test", "Alternator failure", "Low charging voltage under load supports an alternator fault."),
    "wheel_imbalance": ("run_acceleration_test", "Wheel imbalance", "Speed-correlated wheel vibration supports wheel imbalance."),
    "fuel_air_mixture": ("run_mixture_test", "Fuel-air mixture fault", "Lean mixture and elevated fuel use support a mixture fault."),
    "normal": ("run_engine_load_test", "No simulated fault detected", "The engine response remains within expected ranges."),
}


class DemoDecisionProvider:
    name = "demo_simulation"
    model_name = "deterministic-sequence"

    def __init__(self, scenario: str = "misfire"):
        self.scenario = scenario

    async def decide(self, state: dict[str, Any], tools: list[dict[str, Any]]) -> AgentDecision:
        obs = state["observations"]
        test_name, diagnosis, reason = _TESTS.get(self.scenario, _TESTS["misfire"])
        if not any(item["tool"] == "inspect_recent_telemetry" for item in obs):
            action, why = "inspect_recent_telemetry", "Review recent telemetry samples for a persistent signal pattern."
        elif not any(item["kind"] == "diagnostic_test" for item in obs):
            action, why = test_name, f"Run a targeted test to evaluate the leading {diagnosis.casefold()} hypothesis."
        elif not any(item["tool"] == "run_mixture_test" for item in obs) and self.scenario == "misfire":
            action, why = "run_mixture_test", "Check mixture data as a second test to distinguish a misfire from a lean condition."
        else:
            test_obs = [item for item in obs if item["kind"] == "diagnostic_test"]
            evidence = [item["id"] for item in test_obs]
            return AgentDecision.model_validate({
                "action": "finish", "args": {}, "reason": reason,
                "hypotheses": [{"name": diagnosis, "confidence": .82, "status": "supported",
                                "supporting_evidence": evidence, "contradicting_evidence": []}],
                "need_more_information": False,
                "diagnosis": {"name": diagnosis, "confidence": .82, "evidence": evidence,
                              "next_action": "Confirm with an in-person inspection and manufacturer-specific diagnostic procedure."},
            })
        return AgentDecision.model_validate({
            "action": action, "args": {}, "reason": why,
            "hypotheses": [], "need_more_information": True, "diagnosis": None,
        })

    async def aclose(self) -> None:
        return None

    async def answer_question(self, question: str, context: dict[str, Any], knowledge: list[dict[str, Any]]) -> str:
        if knowledge:
            answer = " ".join(item["text"] for item in knowledge[:2])
        else:
            answer = ("I can explain common vehicle symptoms, warning indicators, and maintenance concepts. "
                      "Share a diagnostic code or symptom, or use Investigate to run the simulated diagnostic sequence.")
        if context.get("diagnosis"):
            answer += f" Current simulated investigation: {context['diagnosis']} ({round((context.get('confidence') or 0) * 100)}% confidence)."
        return answer
