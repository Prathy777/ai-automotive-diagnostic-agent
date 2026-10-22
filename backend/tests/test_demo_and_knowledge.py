import asyncio
import os
import unittest
from unittest.mock import patch

from app.agent import investigate
from app.knowledge import search_knowledge
from app.main import ChatRequest, InvestigationRequest, chat, start


class DemoFlowTests(unittest.TestCase):
    def test_demo_runs_real_tools_and_finishes_with_referenced_evidence(self):
        with patch.dict(os.environ, {"DEMO_MODE": "true"}, clear=False):
            result = asyncio.run(investigate("misfire", 8))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["model_provider"], "demo_simulation")
        self.assertEqual(result["tools_used"], ["inspect_recent_telemetry", "run_2000_rpm_test", "run_mixture_test"])
        self.assertEqual(result["diagnosis"], "Engine misfire")
        self.assertGreaterEqual(len(result["observations"]), 4)
        ids = {item["id"] for item in result["observations"]}
        self.assertTrue(set(result["diagnosis_details"]["evidence"]).issubset(ids))

    def test_demo_mode_is_explicit_and_disabled_mode_does_not_use_demo(self):
        with patch.dict(os.environ, {"DEMO_MODE": "false", "MODEL_PROVIDER": "offline", "GEMMA_API_KEY": ""}, clear=False):
            result = asyncio.run(investigate("misfire", 1))
        self.assertEqual(result["model_provider"], "offline")
        self.assertEqual(result["status"], "error")

    def test_vehicle_knowledge_search(self):
        dtc = search_knowledge("P0301")
        self.assertEqual(dtc[0]["id"], "dtc-p0301")
        self.assertEqual([item["id"] for item in search_knowledge("What does P0301 mean?")], ["dtc-p0301"])
        vibration = search_knowledge("engine vibrating")
        self.assertTrue(any(item["id"] == "symptom-vibration" for item in vibration))

    def test_chatbot_to_investigation_handoff_uses_the_demo_harness(self):
        async def run_flow():
            answer = await chat(ChatRequest(message="Investigate this engine vibration"))
            result = await start(InvestigationRequest(scenario="misfire", max_steps=8))
            return answer, result

        with patch.dict(os.environ, {"MODEL_PROVIDER": "demo", "DEMO_MODE": "true"}, clear=False):
            answer, result = asyncio.run(run_flow())
        self.assertTrue(answer["answer"])
        self.assertTrue(answer["suggest_investigation"])
        self.assertEqual(result["model_provider"], "demo_simulation")
        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
