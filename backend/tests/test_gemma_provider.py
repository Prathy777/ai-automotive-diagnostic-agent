import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError

from pydantic import ValidationError

from app.agent import investigate
from app.agent_schemas import AgentDecision, Diagnosis, Hypothesis
from app.providers.base import ProviderConfigurationError, ProviderRequestError
from app.providers.factory import create_provider, provider_health
from app.providers.gemma import DEFAULT_API_BASE_URL, GemmaProvider
from app.simulator import Vehicle
from app.tools.registry import TOOLS, tool_descriptions


def decision_dict(**overrides):
    value = {"action": "read_engine_rpm", "args": {}, "reason": "Check engine-speed stability.",
             "hypotheses": [], "need_more_information": True, "diagnosis": None}
    value.update(overrides)
    return value


def response_for(value=None):
    payload = {"candidates": [{"content": {"parts": [{"text": json.dumps(value or decision_dict())}]}}]}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self): return json.dumps(payload).encode()

    return Response()


class GemmaProviderTests(unittest.TestCase):
    def test_provider_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ProviderConfigurationError, "GEMMA_API_KEY"):
                GemmaProvider()

    def test_factory_explicitly_selects_gemma_and_rejects_unknown_provider(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "gemma", "GEMMA_API_KEY": "test-secret",
                                     "GEMMA_MODEL": "gemma-4-31b-it"}, clear=True):
            self.assertIsInstance(create_provider(), GemmaProvider)
            with self.assertRaisesRegex(ProviderConfigurationError, "Unsupported MODEL_PROVIDER"):
                create_provider("offline")

    def test_gemma_is_the_default_and_only_real_provider(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "gemma", "GEMMA_API_KEY": "test-secret"}, clear=True):
            self.assertIsInstance(create_provider(), GemmaProvider)
        with patch.dict(os.environ, {"MODEL_PROVIDER": "remote", "GEMMA_API_KEY": "test-secret"}, clear=True):
            with self.assertRaisesRegex(ProviderConfigurationError, "Choose gemma or demo"):
                create_provider()

    def test_provider_health_reports_only_verified_gemma_availability(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "gemma", "MODEL_NAME": "gemma-4-31b-it",
                                     "GEMMA_API_KEY": "unit-test-secret", "DEMO_MODE": "false"}, clear=True):
            with patch("app.providers.gemma.GemmaProvider.health_check",
                       new=AsyncMock(return_value={"provider": "gemma", "model": "gemma-4-31b-it",
                                                   "status": "available"})):
                status = asyncio.run(provider_health())
        self.assertEqual(status["provider_status"], "connected")
        self.assertNotIn("unit-test-secret", str(status))

    def test_provider_health_marks_missing_key_unavailable(self):
        with patch.dict(os.environ, {"MODEL_PROVIDER": "gemma", "GEMMA_API_KEY": ""}, clear=True):
            status = asyncio.run(provider_health())
        self.assertEqual(status["provider_status"], "unavailable")
        self.assertEqual(status["api_status"], "missing_credentials")

    def test_gemma_health_check_verifies_model_with_redacted_header(self):
        requests = []

        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self): return b'{"name":"models/gemma-4-31b-it"}'

        def fake_urlopen(request, timeout, context=None):
            requests.append((request, timeout, context))
            return Response()

        provider = GemmaProvider(api_key="unit-test-secret", model_name="gemma-4-31b-it")
        with patch("app.providers.gemma.urlopen", fake_urlopen):
            status = asyncio.run(provider.health_check())
        request, _timeout, context = requests[0]
        self.assertEqual(status["status"], "available")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, f"{DEFAULT_API_BASE_URL}/models/gemma-4-31b-it")
        self.assertEqual(request.get_header("X-goog-api-key"), "unit-test-secret")
        self.assertIsNotNone(context)
        self.assertNotIn("unit-test-secret", str(status))

    def test_provider_constructs_rest_request_and_parses_json(self):
        requests = []

        def fake_urlopen(request, timeout, context=None):
            requests.append((request, timeout))
            return response_for()

        state = {"initial_telemetry": {"rpm": 812, "battery_voltage": 13.8},
                 "observations": [{"id": "obs_001", "data": {"rpm": 812}}],
                 "hypotheses": [], "previous_actions": [], "step": 1, "remaining_action_budget": 8,
                 "scenario": "hidden-misfire", "expected_diagnosis": "secret label"}
        with patch("app.providers.gemma.urlopen", fake_urlopen):
            result = asyncio.run(GemmaProvider(api_key="unit-test-secret", model_name="gemma-4-31b-it", timeout=4)
                                 .decide(state, tool_descriptions()))

        self.assertIsInstance(result, AgentDecision)
        self.assertEqual(result.action, "read_engine_rpm")
        request, timeout = requests[0]
        self.assertEqual(request.full_url, f"{DEFAULT_API_BASE_URL}/models/gemma-4-31b-it:generateContent")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("X-goog-api-key"), "unit-test-secret")
        self.assertNotIn("unit-test-secret", request.full_url)
        self.assertEqual(timeout, 4)
        body = json.loads(request.data)
        self.assertEqual(body["generationConfig"]["temperature"], 0.1)
        self.assertNotIn("responseMimeType", body["generationConfig"])
        self.assertNotIn("systemInstruction", body)
        prompt = body["contents"][0]["parts"][0]["text"]
        self.assertIn("You are the reasoning model", prompt)
        self.assertIn('"rpm": 812', prompt)
        self.assertNotIn("hidden-misfire", prompt)
        self.assertNotIn("secret label", prompt)

    def test_provider_ignores_thought_parts_when_parsing_decision(self):
        value = {"candidates": [{"content": {"parts": [
            {"text": "Private reasoning that is not JSON.", "thought": True},
            {"text": json.dumps(decision_dict())},
        ]}}]}
        parsed = GemmaProvider._decision_text(value)
        self.assertEqual(json.loads(parsed)["action"], "read_engine_rpm")

    def test_provider_strips_json_markdown_fence(self):
        value = {"candidates": [{"content": {"parts": [
            {"text": "```json\n" + json.dumps(decision_dict()) + "\n```"},
        ]}}]}
        self.assertEqual(json.loads(GemmaProvider._decision_text(value))["action"], "read_engine_rpm")

    def test_provider_extracts_json_from_surrounding_text(self):
        value = {"candidates": [{"content": {"parts": [
            {"text": "Decision follows:\n" + json.dumps(decision_dict()) + "\nEnd."},
        ]}}]}
        self.assertEqual(json.loads(GemmaProvider._decision_text(value))["action"], "read_engine_rpm")

    def test_api_error_is_sanitized(self):
        def fail(_request, **_kwargs):
            raise HTTPError("https://example.invalid", 403, "private api details", {}, None)

        with patch("app.providers.gemma.urlopen", fail):
            with self.assertRaisesRegex(ProviderRequestError, "rejected the configured credentials") as caught:
                asyncio.run(GemmaProvider(api_key="unit-test-secret").decide({}, tool_descriptions()))
        self.assertEqual(caught.exception.code, "authentication_error")
        self.assertNotIn("private api details", str(caught.exception))
        self.assertNotIn("unit-test-secret", str(caught.exception))

    def test_transient_http_errors_are_retried_at_most_three_times(self):
        provider = GemmaProvider(api_key="unit-test-secret")
        unavailable = HTTPError("https://example.invalid", 503, "temporary", {}, None)
        with patch.object(provider, "_request_once", side_effect=[unavailable, unavailable, {"ok": True}]) as request:
            with patch("app.providers.gemma.time.sleep") as sleep:
                self.assertEqual(provider._request({"contents": []}), {"ok": True})
        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_http_500_502_and_timeout_retry_exhaustion_is_bounded(self):
        provider = GemmaProvider(api_key="unit-test-secret")
        failures = [HTTPError("https://example.invalid", status, "temporary", {}, None)
                    for status in (500, 502, 503)] + [TimeoutError("temporary timeout")]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__, status=getattr(failure, "code", None)):
                with patch.object(provider, "_request_once", side_effect=failure) as request:
                    with patch("app.providers.gemma.time.sleep") as sleep:
                        with self.assertRaises(type(failure)):
                            provider._request({"contents": []})
                self.assertEqual(request.call_count, 3)
                self.assertEqual(sleep.call_count, 2)

    def test_mock_gemma_completes_a_full_harness_investigation(self):
        decisions = [
            decision_dict(action="inspect_recent_telemetry", reason="Review recent signals."),
            decision_dict(action="run_2000_rpm_test", reason="Check RPM response."),
            decision_dict(action="run_mixture_test", reason="Compare fuel-air evidence."),
            decision_dict(action="finish", reason="Two targeted tests support this diagnosis.",
                          hypotheses=[{"name": "Engine misfire", "confidence": .84, "status": "supported",
                                       "supporting_evidence": ["obs_003", "obs_004"],
                                       "contradicting_evidence": []}],
                          need_more_information=False,
                          diagnosis={"name": "Engine misfire", "confidence": .84,
                                     "evidence": ["obs_003", "obs_004"],
                                     "next_action": "Confirm with a qualified technician."}),
        ]
        prompts = []

        def mock_response(payload):
            prompts.append(json.dumps(payload))
            value = decisions[len(prompts) - 1]
            return {"candidates": [{"content": {"parts": [{"text": json.dumps(value)}]}}]}

        with patch.dict(os.environ, {"MODEL_PROVIDER": "gemma", "MODEL_NAME": "gemma-4-31b-it",
                                     "GEMMA_API_KEY": "unit-test-secret", "DEMO_MODE": "false"}, clear=True):
            with patch.object(GemmaProvider, "_request_once", side_effect=mock_response):
                result = asyncio.run(investigate("misfire", 8))

        self.assertEqual(len(prompts), 4)
        self.assertTrue(all("unit-test-secret" not in prompt for prompt in prompts))
        self.assertTrue(all('"scenario"' not in prompt and "hidden-misfire" not in prompt for prompt in prompts))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_provider"], "gemma")
        self.assertEqual(result["tools_used"], ["inspect_recent_telemetry", "run_2000_rpm_test", "run_mixture_test"])
        self.assertEqual(result["diagnosis"], "Engine misfire")
        self.assertEqual(result["diagnosis_details"]["evidence"], ["obs_003", "obs_004"])

    def test_diagnostic_request_redacts_api_key_and_prompt(self):
        provider = GemmaProvider(api_key="unit-test-secret")
        payload = GemmaProvider.build_payload({"initial_telemetry": {"rpm": 800}}, tool_descriptions())
        body = json.dumps(payload).encode()
        with patch.dict(os.environ, {"GEMMA_DIAGNOSTICS": "1"}):
            with self.assertLogs("app.providers.gemma", level="INFO") as logs:
                provider._diagnostic_request(payload, body, {
                    "Content-Type": "application/json", "x-goog-api-key": provider.api_key,
                })
        output = "\n".join(logs.output)
        self.assertIn("payload_structure", output)
        self.assertIn("serialized_bytes", output)
        self.assertNotIn("unit-test-secret", output)
        self.assertNotIn("x-goog-api-key", output)

    def test_malformed_gemma_json_is_rejected(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self): return b'{"candidates":[{"content":{"parts":[{"text":"not-json"}]}}]}'

        with patch("app.providers.gemma.urlopen", lambda *_args, **_kwargs: Response()):
            with self.assertRaisesRegex(ProviderRequestError, "invalid structured decision"):
                asyncio.run(GemmaProvider(api_key="test").decide({}, tool_descriptions()))

    def test_confidence_and_diagnosis_schema_are_validated(self):
        with self.assertRaises(ValidationError):
            Hypothesis(name="bad", confidence=1.1, status="leading", supporting_evidence=[], contradicting_evidence=[])
        with self.assertRaises(ValidationError):
            Diagnosis(name="fault", confidence=.8, evidence=["obs_001"])

    def test_simulator_has_deterministic_fault_telemetry(self):
        misfire = Vehicle("misfire").telemetry()
        hot = Vehicle("overheating").telemetry()
        charging = Vehicle("alternator_failure").telemetry()
        wheel = Vehicle("wheel_imbalance").telemetry()
        mixture = Vehicle("fuel_air_mixture").telemetry()
        self.assertGreater(misfire["engine_vibration"], .7)
        self.assertGreater(hot["coolant_temperature"], 100)
        self.assertLess(charging["alternator_output"], 12)
        self.assertGreater(wheel["wheel_vibration"], .8)
        self.assertGreater(mixture["air_fuel_ratio"], 17)
        required = {"rpm", "engine_temperature", "coolant_temperature", "battery_voltage", "throttle_position",
                    "fuel_consumption", "engine_vibration", "vehicle_speed", "acceleration", "intake_pressure",
                    "air_fuel_ratio", "electrical_load"}
        self.assertTrue(required.issubset(misfire))

    def test_tool_registry_exposes_diagnostic_actions(self):
        expected = {"read_engine_rpm", "read_engine_temperature", "read_battery_voltage", "read_engine_vibration",
                    "read_fuel_consumption", "read_vehicle_speed", "inspect_recent_telemetry",
                    "run_acceleration_test", "run_electrical_load_test", "run_cooling_system_test", "run_engine_load_test"}
        self.assertTrue(expected.issubset(TOOLS))
        self.assertTrue(all(item["arguments"]["additionalProperties"] is False for item in tool_descriptions()))


class StaticProvider:
    name = "test-provider"
    model_name = "test-model"

    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.contexts = []

    async def decide(self, state, _tools):
        self.contexts.append(state)
        return self.decisions.pop(0)


def decision(action, *, hypotheses=None, need_more=True, diagnosis=None):
    return AgentDecision(action=action, args={}, reason="This observation can distinguish candidate faults.",
                         hypotheses=hypotheses or [], need_more_information=need_more, diagnosis=diagnosis)


class AgentHarnessTests(unittest.TestCase):
    def test_invalid_action_is_rejected(self):
        result = asyncio.run(investigate("misfire", model=StaticProvider(decision("run_shell"))))
        self.assertEqual(result["stopping_reason"], "invalid_action")

    def test_invalid_evidence_reference_is_rejected(self):
        hypothesis = Hypothesis(name="misfire", confidence=.7, status="leading",
                                supporting_evidence=["obs_999"], contradicting_evidence=[])
        result = asyncio.run(investigate("misfire", model=StaticProvider(
            decision("read_engine_rpm", hypotheses=[hypothesis]))))
        self.assertEqual(result["stopping_reason"], "invalid_evidence_reference")

    def test_duplicate_tool_call_is_rejected(self):
        result = asyncio.run(investigate("misfire", model=StaticProvider(
            decision("read_engine_rpm"), decision("read_engine_rpm"))))
        self.assertEqual(result["stopping_reason"], "repeated_action")
        self.assertEqual(result["tools_used"], ["read_engine_rpm"])

    def test_initial_telemetry_is_sent_without_scenario_label(self):
        model = StaticProvider(decision("read_engine_rpm"), decision("read_engine_rpm"))
        asyncio.run(investigate("misfire", model=model))
        state = model.contexts[0]
        self.assertTrue(state["initial_telemetry"]["rpm"])
        self.assertEqual(state["observations"][0]["id"], "obs_001")
        self.assertNotIn("scenario", state)
        self.assertNotIn("fault", state)

    def test_finishes_with_model_evidence_and_recommended_action(self):
        hypothesis = Hypothesis(name="engine misfire", confidence=.73, status="supported",
                                supporting_evidence=["obs_004"], contradicting_evidence=[])
        diagnosis = Diagnosis(name="engine misfire", confidence=.78, evidence=["obs_004"],
                              next_action="Inspect ignition components and verify each cylinder.")
        model = StaticProvider(
            decision("read_engine_rpm"), decision("read_engine_vibration"), decision("run_engine_load_test"),
            decision("finish", hypotheses=[hypothesis], need_more=False, diagnosis=diagnosis))
        result = asyncio.run(investigate("misfire", model=model))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["diagnosis"], "engine misfire")
        self.assertEqual(result["confidence"], .78)
        self.assertTrue(result["next_action"].startswith("Inspect ignition"))
        self.assertEqual(result["tools_used"], ["read_engine_rpm", "read_engine_vibration", "run_engine_load_test"])
        self.assertEqual(result["tool_calls"][-1]["observation_id"], "obs_004")
        self.assertEqual(result["diagnosis_details"]["evidence"], ["obs_004"])

    def test_harness_stops_after_eight_model_turns(self):
        actions = ["read_engine_rpm", "read_engine_temperature", "read_battery_voltage", "read_engine_vibration",
                   "read_wheel_vibration", "read_fuel_consumption", "read_vehicle_speed", "read_air_fuel_ratio"]
        model = StaticProvider(*(decision(action) for action in actions))
        result = asyncio.run(investigate("misfire", max_steps=8, model=model))
        self.assertEqual(result["stopping_reason"], "max_steps")
        self.assertEqual(len(model.contexts), 8)
        with self.assertRaises(ValueError):
            asyncio.run(investigate("misfire", max_steps=9, model=model))

    def test_duplicate_action_can_stop_in_a_short_run(self):
        result = asyncio.run(investigate("misfire", max_steps=8, model=StaticProvider(
            decision("read_engine_rpm"), decision("read_engine_rpm"))))
        self.assertEqual(result["stopping_reason"], "repeated_action")


if __name__ == "__main__":
    unittest.main()
