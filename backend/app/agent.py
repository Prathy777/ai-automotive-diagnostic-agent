"""Bounded provider-agnostic diagnostic harness; local code validates and executes tools."""
import logging
import os
import time
import uuid
from copy import deepcopy
from typing import Any

from pydantic import ValidationError

from .agent_schemas import AgentDecision, Diagnosis, Hypothesis
from .providers.base import ProviderConfigurationError, ProviderRequestError
from .simulator import Vehicle
from .tools.registry import TOOLS, tool_descriptions

logger = logging.getLogger(__name__)
MAX_INVESTIGATION_STEPS = 8


async def investigate(scenario: str, max_steps: int = MAX_INVESTIGATION_STEPS,
                      model: Any | None = None) -> dict[str, Any]:
    """Investigate one simulated fault without ever sending its label to the model."""
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or not 1 <= max_steps <= MAX_INVESTIGATION_STEPS:
        raise ValueError(f"max_steps must be between 1 and {MAX_INVESTIGATION_STEPS}")

    session_id = str(uuid.uuid4())
    vehicle = Vehicle(fault=scenario)
    events: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    hypotheses: list[Hypothesis] = []
    diagnosis: Diagnosis | None = None
    status = "incomplete"
    stopping_reason = "max_steps"
    from .providers.factory import provider_settings
    settings = provider_settings()
    selected_provider = str(settings["model_provider"])
    demo_mode = bool(settings["demo_mode"])
    provider_name = "demo_simulation" if demo_mode and model is None else selected_provider
    model_name = str(settings["model_name"])

    def emit(kind: str, **data: Any) -> None:
        events.append({"type": kind, "session_id": session_id, **data})

    def result() -> dict[str, Any]:
        return {
            "session_id": session_id,
            "scenario": scenario,
            "status": status,
            "diagnosis": diagnosis.name if diagnosis else None,
            "confidence": diagnosis.confidence if diagnosis else None,
            "diagnosis_details": diagnosis.model_dump() if diagnosis else None,
            "next_action": diagnosis.next_action if diagnosis else None,
            "observations": observations,
            "tools_used": [call["tool"] for call in tool_calls],
            "tool_calls": tool_calls,
            "executed_tests": [call["tool"] for call in tool_calls if call["kind"] == "diagnostic_test"],
            "investigation_steps": actions,
            "hypotheses": [hypothesis.model_dump() for hypothesis in hypotheses],
            "events": events,
            "stopping_reason": stopping_reason,
            "model_provider": provider_name,
            "model_name": model_name,
        }

    if model is None:
        try:
            from .providers.factory import create_provider
            model = create_provider("demo" if demo_mode else selected_provider, scenario)
        except ProviderConfigurationError as exc:
            stopping_reason, status = "provider_configuration_error", "error"
            emit("agent.started", provider=provider_name, model=model_name)
            emit("agent.error", reason=stopping_reason, message=str(exc))
            return result()

    provider_name, model_name = model.name, model.model_name
    emit("agent.started", provider=provider_name, model=model_name)

    initial_telemetry = vehicle.telemetry()
    initial_observation = {
        "id": "obs_001", "kind": "initial_telemetry", "tool": "observe_vehicle", "data": initial_telemetry,
    }
    observations.append(initial_observation)
    emit("telemetry.observed", observation=initial_observation)
    tools = tool_descriptions()

    for step in range(1, max_steps + 1):
        # This public-state allowlist intentionally excludes scenario, fault, and expected diagnosis.
        context = {
            "initial_telemetry": initial_telemetry,
            "observations": observations,
            "hypotheses": [hypothesis.model_dump() for hypothesis in hypotheses],
            "previous_actions": actions,
            "step": step,
            "remaining_action_budget": max_steps - len(tool_calls),
        }
        decision_started = time.perf_counter()
        try:
            decision = await model.decide(deepcopy(context), deepcopy(tools))
            if not isinstance(decision, AgentDecision):
                decision = AgentDecision.model_validate(decision)
        except ProviderConfigurationError as exc:
            stopping_reason, status = "provider_configuration_error", "error"
            emit("agent.error", reason=stopping_reason, message=str(exc))
            break
        except ProviderRequestError as exc:
            stopping_reason, status = exc.code, "error"
            emit("agent.error", reason=exc.code, message=str(exc))
            break
        except (ValidationError, ValueError, TypeError):
            stopping_reason, status = "invalid_structured_response", "error"
            emit("agent.error", reason=stopping_reason, message="Provider returned an invalid decision")
            break
        except Exception as exc:
            stopping_reason, status = "provider_error", "error"
            emit("agent.error", reason=stopping_reason, message=f"{provider_name} provider unavailable")
            logger.error("agent_provider_error provider=%s model=%s session_id=%s step=%s error_type=%s latency_ms=%.1f",
                         provider_name, model_name, session_id, step, type(exc).__name__,
                         (time.perf_counter() - decision_started) * 1000)
            break

        available_ids = {observation["id"] for observation in observations}
        referenced_ids = set()
        for hypothesis in decision.hypotheses:
            referenced_ids.update(hypothesis.supporting_evidence)
            referenced_ids.update(hypothesis.contradicting_evidence)
        if decision.diagnosis:
            referenced_ids.update(decision.diagnosis.evidence)
        if not referenced_ids.issubset(available_ids):
            stopping_reason, status = "invalid_evidence_reference", "error"
            emit("agent.error", reason=stopping_reason,
                 message="Model referenced an observation that does not exist")
            break
        hypotheses = decision.hypotheses or hypotheses

        if decision.action == "finish":
            test_observation_ids = {item["id"] for item in observations if item["kind"] == "diagnostic_test"}
            sufficient = (
                not decision.need_more_information
                and len(observations) >= 3
                and bool(test_observation_ids)
                and bool(hypotheses)
                and decision.diagnosis is not None
                and bool(test_observation_ids.intersection(decision.diagnosis.evidence))
                and any(hypothesis.name.casefold() in decision.diagnosis.name.casefold()
                        or decision.diagnosis.name.casefold() in hypothesis.name.casefold()
                        for hypothesis in hypotheses)
            )
            if not sufficient:
                reason = "Need more evidence and a targeted test before providing a diagnosis."
                actions.append({"step": step, "action": "finish", "reason": reason, "status": "rejected"})
                emit("agent.finish_rejected", reason=reason, step=step)
                continue
            diagnosis = decision.diagnosis
            stopping_reason, status = "evidence_sufficient", "completed"
            actions.append({"step": step, "action": "finish", "reason": decision.reason, "status": "accepted"})
            emit("diagnosis.generated", diagnosis=diagnosis.model_dump(), reason=decision.reason)
            break

        tool = TOOLS.get(decision.action)
        if tool is None:
            stopping_reason, status = "invalid_action", "error"
            emit("agent.error", reason=stopping_reason, message="Requested action is not registered")
            break
        if decision.args != {}:
            stopping_reason, status = "invalid_tool_arguments", "error"
            emit("agent.error", reason=stopping_reason, message="Tool does not accept the supplied arguments")
            break
        if any(call["tool"] == decision.action for call in tool_calls):
            stopping_reason, status = "repeated_action", "error"
            emit("agent.error", reason=stopping_reason, message="Repeated diagnostic action rejected")
            break

        tool_id = f"tool_{len(tool_calls) + 1:03d}"
        action = {"step": step, "tool_id": tool_id, "action": decision.action,
                  "args": decision.args, "reason": decision.reason, "kind": tool.kind, "status": "running"}
        actions.append(action)
        emit("agent.decision", tool_id=tool_id, action=decision.action, args=decision.args,
             reason=decision.reason, step=step)
        emit("tool.started", tool_id=tool_id, tool=decision.action, step=step)
        try:
            value = tool.handler(vehicle, decision.args)
        except Exception as exc:
            stopping_reason, status = "diagnostic_tool_error", "error"
            action["status"] = "error"
            emit("agent.error", reason=stopping_reason, message="A local diagnostic tool failed")
            logger.error("tool_execution_failed session_id=%s step=%s tool=%s error_type=%s",
                         session_id, step, decision.action, type(exc).__name__)
            break
        observation = {"id": f"obs_{len(observations) + 1:03d}", "tool_id": tool_id,
                       "kind": tool.kind, "tool": decision.action, "data": value}
        observations.append(observation)
        tool_calls.append({"tool_id": tool_id, "tool": decision.action, "kind": tool.kind,
                           "observation_id": observation["id"], "result": value})
        action["status"] = "completed"
        action["observation_id"] = observation["id"]
        emit("tool.completed", tool_id=tool_id, tool=decision.action, observation=observation)
        if decision.hypotheses:
            emit("hypothesis.updated", hypotheses=[item.model_dump() for item in hypotheses],
                 based_on=observation["id"])

    else:
        stopping_reason, status = "max_steps", "incomplete"

    close_provider = getattr(model, "aclose", None)
    if close_provider is not None:
        try:
            await close_provider()
        except Exception:
            logger.warning("provider_close_failed provider=%s model=%s session_id=%s",
                           provider_name, model_name, session_id)
    return result()
