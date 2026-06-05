"""Pydantic v2 contracts shared by all four cognitive roles.

Every boundary between roles is a class in this file. Editing a class here
ripples through Memory, Perception, Decision, Action and the agent loop —
that's the point: one source of truth.
"""
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class MemoryItem(BaseModel):
    """One row in Memory. Discriminated by ``kind``."""
    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str] = Field(default_factory=list)
    descriptor: str
    value: dict[str, Any] = Field(default_factory=dict)
    artifact_id: Optional[str] = None
    source: str
    run_id: str
    goal_id: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Artifact(BaseModel):
    """Metadata for one stored blob in the ArtifactStore."""
    id: str
    content_type: str
    size_bytes: int
    source: str
    descriptor: str


class Goal(BaseModel):
    """One bounded sub-task emitted by Perception."""
    id: str
    text: str
    done: bool
    attach_artifact_id: Optional[str]


class Observation(BaseModel):
    """Perception's typed output: the current goal list."""
    goals: list[Goal] = Field(default_factory=list)

    def all_done(self) -> bool:
        return bool(self.goals) and all(g.done for g in self.goals)

    def next_unfinished(self) -> Optional[Goal]:
        for g in self.goals:
            if not g.done:
                return g
        return None


class ToolCall(BaseModel):
    """One tool dispatch request emitted by Decision."""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class DecisionOutput(BaseModel):
    """Decision's typed output: exactly one of ``answer`` or ``tool_call``."""
    answer: Optional[str] = None
    tool_call: Optional[ToolCall] = None

    @model_validator(mode="after")
    def exactly_one(self) -> "DecisionOutput":
        if (self.answer is None) == (self.tool_call is None):
            raise ValueError(
                "DecisionOutput must populate exactly one of "
                "`answer` or `tool_call`."
            )
        return self

    @property
    def is_answer(self) -> bool:
        return self.answer is not None

    @property
    def is_tool_call(self) -> bool:
        return self.tool_call is not None
