"""FastAPI interface for the MLOps agent: validation in, structured response out."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from agent.llm import OllamaClient
from agent.orchestrator import Agent
from agent.schemas import AgentResponse, ChatRequest, ConfirmRequest, ToolSpec
from agent.tools.base import get_registry


def create_app(agent: Agent | None = None) -> FastAPI:
    app = FastAPI(title="MLOps AI Agent", version="1.0.0")
    state = {"agent": agent}

    def get_agent() -> Agent:
        if state["agent"] is None:
            state["agent"] = Agent(llm=OllamaClient())
        return state["agent"]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/tools", response_model=list[ToolSpec])
    def tools() -> list[ToolSpec]:
        return [t.spec() for t in get_registry().values()]

    @app.post("/chat", response_model=AgentResponse)
    def chat(req: ChatRequest) -> AgentResponse:
        if not req.message.strip():
            raise HTTPException(status_code=422, detail="message must not be blank")
        return get_agent().handle(req.message)

    @app.post("/confirm", response_model=AgentResponse)
    def confirm(req: ConfirmRequest) -> AgentResponse:
        return get_agent().confirm(req.action_id)

    @app.post("/reject")
    def reject(req: ConfirmRequest) -> dict[str, bool]:
        return {"rejected": get_agent().reject(req.action_id)}

    return app


app = create_app()
