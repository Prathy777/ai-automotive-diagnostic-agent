from typing import Any, Protocol

from ..agent_schemas import AgentDecision


class ModelProvider(Protocol):
    name: str
    model_name: str

    async def decide(self, state: dict[str, Any], tools: list[dict[str, Any]]) -> AgentDecision: ...
    async def answer_question(self, question: str, context: dict[str, Any], knowledge: list[dict[str, Any]]) -> str: ...
    async def health_check(self) -> dict[str, Any]: ...
    async def aclose(self) -> None: ...


class ProviderConfigurationError(RuntimeError):
    """The selected provider cannot be configured from the current environment."""


class ProviderRequestError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
