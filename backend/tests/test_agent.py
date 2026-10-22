import asyncio
import unittest

from app.agent import investigate
from app.agent_schemas import AgentDecision


class NoFinishProvider:
    name = "test-provider"
    model_name = "test-model"

    async def decide(self, _state, _tools):
        return AgentDecision(action="read_engine_rpm", args={}, reason="Inspect engine speed first.",
                             hypotheses=[], need_more_information=True, diagnosis=None)


class AgentTests(unittest.TestCase):
    def test_repeated_call_guard(self):
        result = asyncio.run(investigate("misfire", max_steps=8, model=NoFinishProvider()))
        self.assertEqual(result["stopping_reason"], "repeated_action")
        self.assertEqual(len(result["tools_used"]), 1)
        self.assertEqual(result["tools_used"], ["read_engine_rpm"])
