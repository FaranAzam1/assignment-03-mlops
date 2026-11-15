"""Pydantic models shared by the agent, the tools and the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ToolStatus = Literal["ok", "error", "timeout", "denied", "invalid", "unavailable", "pending"]


class ToolResult(BaseModel):
    """Outcome of one tool call. `data` is only ever filled from a real tool response."""

    tool: str
    status: ToolStatus
    data: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    request_id: str = ""
    args: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class PendingAction(BaseModel):
    action_id: str
    tool: str
    args: dict[str, Any]
    description: str
    request_id: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ConfirmRequest(BaseModel):
    action_id: str = Field(min_length=4, max_length=64)


class AgentResponse(BaseModel):
    request_id: str
    answer: str
    mode: str
    tool_calls: list[ToolResult] = Field(default_factory=list)
    pending_actions: list[PendingAction] = Field(default_factory=list)


class ToolSpec(BaseModel):
    name: str
    module: str
    description: str
    mutating: bool
    parameters: dict[str, Any]
