"""Send the exact first-turn Gemma investigation request directly to Google."""
import json
import logging
import os
import sys
from urllib.parse import quote

from app.providers.gemma import DEFAULT_API_BASE_URL, DEFAULT_MODEL_NAME, GemmaProvider, SYSTEM_INSTRUCTION
from app.simulator import Vehicle
from app.tools.registry import tool_descriptions
from gemma_progressive_test import load_dotenv, send


def main() -> int:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.environ["GEMMA_DIAGNOSTICS"] = "1"
    model = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    if model != "gemma-4-31b-it":
        raise SystemExit("Set MODEL_NAME=gemma-4-31b-it before running this reproduction.")
    provider = GemmaProvider(model_name=model, timeout=float(os.getenv("MODEL_TIMEOUT", "60")))
    base_url = os.getenv("GEMMA_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")
    provider.endpoint = f"{base_url}/models/{quote(model, safe='')}:generateContent"

    # The hidden scenario is used only by the local digital twin to derive telemetry.
    # It is intentionally absent from the model context passed to build_payload.
    vehicle = Vehicle(fault="misfire")
    initial_telemetry = vehicle.telemetry()
    initial_observation = {"id": "obs_001", "kind": "initial_telemetry",
                           "tool": "observe_vehicle", "data": initial_telemetry}
    state = {
        "initial_telemetry": initial_telemetry,
        "observations": [initial_observation],
        "hypotheses": [],
        "previous_actions": [],
        "step": 1,
        "remaining_action_budget": 8,
    }
    if "--after-two-tests" in sys.argv:
        first_reason = "The initial telemetry shows an air-fuel ratio of 16.2, which is lean. I need to run a mixture test to further investigate the fuel-air balance and intake pressure."
        mixture = vehicle.run_test("run_mixture_test")
        second_reason = "The mixture test indicated that the air-fuel ratio is within the expected range despite the initial reading of 16.2. I will now test the engine under load to see if any performance or mixture issues manifest during operation."
        load_test = vehicle.run_test("run_engine_load_test")
        state = {
            "initial_telemetry": initial_telemetry,
            "observations": [
                initial_observation,
                {"id": "obs_002", "tool_id": "tool_001", "kind": "diagnostic_test",
                 "tool": "run_mixture_test", "data": mixture},
                {"id": "obs_003", "tool_id": "tool_002", "kind": "diagnostic_test",
                 "tool": "run_engine_load_test", "data": load_test},
            ],
            "hypotheses": [
                {"name": "Lean fuel mixture", "confidence": 0.1, "status": "ruled_out",
                 "supporting_evidence": ["obs_001"], "contradicting_evidence": ["obs_002"]},
                {"name": "Engine performance issue under load", "confidence": 0.3,
                 "status": "leading", "supporting_evidence": [], "contradicting_evidence": []},
            ],
            "previous_actions": [
                {"step": 1, "tool_id": "tool_001", "action": "run_mixture_test", "args": {},
                 "reason": first_reason, "kind": "diagnostic_test", "status": "completed",
                 "observation_id": "obs_002"},
                {"step": 2, "tool_id": "tool_002", "action": "run_engine_load_test", "args": {},
                 "reason": second_reason, "kind": "diagnostic_test", "status": "completed",
                 "observation_id": "obs_003"},
            ],
            "step": 3,
            "remaining_action_budget": 6,
        }
    payload = GemmaProvider.build_payload(state, tool_descriptions())
    if "--plain-text-instructions" in sys.argv:
        user_text = SYSTEM_INSTRUCTION + "\n\n" + payload["contents"][0]["parts"][0]["text"]
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {"temperature": 0.1},
        }
        print("Request variant: instructions merged into user text; no systemInstruction or responseMimeType")
    print("Request payload summary:")
    print(json.dumps(GemmaProvider._payload_summary(payload, provider.api_key), sort_keys=True))
    result = send(payload, provider)
    return 0 if isinstance(result["status"], int) and 200 <= result["status"] < 300 else 1


if __name__ == "__main__":
    raise SystemExit(main())
