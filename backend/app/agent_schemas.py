"""Validated boundary types shared by providers and the agent harness."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr


class Hypothesis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: StrictStr = Field(min_length=1, max_length=120)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)
    status: Literal["leading", "supported", "confirmed", "ruled_out"]
    supporting_evidence: list[StrictStr]
    contradicting_evidence: list[StrictStr]


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: StrictStr = Field(min_length=1, max_length=160)
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False, strict=True)
    evidence: list[StrictStr] = Field(min_length=1)
    next_action: StrictStr = Field(min_length=3, max_length=500)


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: StrictStr = Field(min_length=1, max_length=100)
    args: dict
    reason: StrictStr = Field(min_length=3, max_length=500)
    hypotheses: list[Hypothesis]
    need_more_information: StrictBool
    diagnosis: Diagnosis | None
