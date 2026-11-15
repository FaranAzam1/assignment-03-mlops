"""The agent loop: choose tools, execute them under permission rules, report from evidence."""

from __future__ import annotations

import json
import uuid
from typing import Any

from agent import planner
from agent.config import Settings, get_settings
from agent.llm import LLM, LLMError
from agent.logging_utils import log_event
from agent.reporting import build_report
from agent.schemas import AgentResponse, PendingAction, ToolResult
from agent.tools.base import Tool, execute_tool, get_registry

MAX_LLM_ROUNDS = 3
MAX_TOOL_CALLS = 8

SYSTEM_PROMPT = (
    "You are an MLOps operations agent. You have tools for DVC (datasets), MLflow (experiments), "
    "Airflow (training pipeline) and Kubernetes (deployment). For any question about datasets, "
    "runs, metrics, pipelines or deployments you MUST call the relevant tool(s); never answer "
    "from memory and never guess values. Call only tools that are needed. Do not trigger "
    "retraining, restarts or scaling unless the user explicitly asks."
)


class Agent:
    def __init__(
        self,
        llm: LLM | None = None,
        settings: Settings | None = None,
        registry: dict[str, Tool] | None = None,
        llm_summary: bool | None = None,
    ):
        self.llm = llm
        self.settings = settings or get_settings()
        self.registry = registry if registry is not None else get_registry()
        # Off by default: a 1B model's prose summary was observed to misstate correct evidence.
        self.llm_summary = self.settings.llm_summary if llm_summary is None else llm_summary
        self._pending: dict[str, PendingAction] = {}

    # ------------------------------------------------------------------ tool dispatch
    def _run_tool(self, name: str, args: dict[str, Any], request_id: str) -> ToolResult:
        """Execute a tool. Mutating tools are first attempted unconfirmed: the tool itself
        decides (namespace / DAG allow-list) and, if only confirmation is missing, we park
        the action until the user confirms it explicitly."""
        tool = self.registry.get(name)
        result = execute_tool(name, args, request_id, confirmed=False,
                              settings=self.settings, registry=self.registry)
        if tool is not None and tool.mutating and result.status == "denied" \
                and "explicit user confirmation" in (result.error or ""):
            action = PendingAction(
                action_id=uuid.uuid4().hex[:8],
                tool=name,
                args=args,
                description=f"{name} with {json.dumps(args)}",
                request_id=request_id,
            )
            self._pending[action.action_id] = action
            result = result.model_copy(update={"status": "pending",
                                               "error": f"awaiting confirmation (action_id={action.action_id})"})
        return result

    def confirm(self, action_id: str) -> AgentResponse:
        action = self._pending.pop(action_id, None)
        request_id = uuid.uuid4().hex[:12]
        if action is None:
            result = ToolResult(tool="confirm", status="invalid", error=f"unknown action_id '{action_id}'",
                                request_id=request_id)
            log_event(self.settings.log_file, "confirm", request_id=request_id, status="invalid",
                      action_id=action_id)
            return AgentResponse(request_id=request_id, answer=f"No pending action with id '{action_id}'.",
                                 mode="confirm", tool_calls=[result])
        log_event(self.settings.log_file, "confirm", request_id=request_id, status="accepted",
                  action_id=action_id, tool=action.tool)
        result = execute_tool(action.tool, action.args, request_id, confirmed=True,
                              settings=self.settings, registry=self.registry)
        return AgentResponse(request_id=request_id, answer=build_report(action.description, [result]),
                             mode="confirm", tool_calls=[result])

    def reject(self, action_id: str) -> bool:
        return self._pending.pop(action_id, None) is not None

    # ------------------------------------------------------------------ strategies
    def _tools_for(self, message: str) -> list[Tool]:
        mods = planner.modules_in(message)
        tools = [t for t in self.registry.values() if not mods or t.module in mods]
        return tools

    def _run_with_llm(self, message: str, request_id: str) -> list[ToolResult] | None:
        """Let the LLM choose tools. Returns None if the LLM failed or chose nothing."""
        assert self.llm is not None
        tools = [t.llm_schema() for t in self._tools_for(message)]
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": message}]
        evidence: list[ToolResult] = []
        for _ in range(MAX_LLM_ROUNDS):
            try:
                resp = self.llm.chat(messages, tools)
            except LLMError as exc:
                log_event(self.settings.log_file, "llm_error", request_id=request_id, error=str(exc))
                return evidence or None
            if not resp.tool_calls:
                break
            messages.append({"role": "assistant", "content": resp.content or "",
                             "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                            for c in resp.tool_calls]})
            for call in resp.tool_calls:
                if len(evidence) >= MAX_TOOL_CALLS:
                    break
                args = {k: v for k, v in (call["arguments"] or {}).items() if not k.startswith("_")}
                result = self._run_tool(call["name"], args, request_id)
                evidence.append(result)
                messages.append({"role": "tool", "tool_name": call["name"],
                                 "content": json.dumps(result.model_dump(include={"status", "data", "error"}),
                                                       default=str)[:3000]})
        return evidence or None

    def _run_plan(self, steps: list[planner.Step], request_id: str) -> list[ToolResult]:
        evidence: list[ToolResult] = []
        for step in steps[:MAX_TOOL_CALLS + 4]:
            args = step.resolve(evidence)
            if args is None:  # dependency produced nothing (e.g. no failed task to diagnose)
                continue
            evidence.append(self._run_tool(step.tool, args, request_id))
        return evidence

    def _summarise(self, message: str, evidence: list[ToolResult], request_id: str) -> str | None:
        # A small model summarising partial evidence tends to paper over the gaps: only summarise
        # when every tool call succeeded. Otherwise the structured report alone speaks.
        if not (self.llm and self.llm_summary) or not evidence or not all(r.ok for r in evidence):
            return None
        blob = json.dumps([r.model_dump(include={"tool", "status", "data", "error"}) for r in evidence],
                          default=str)[:6000]
        prompt = ("Summarise in at most 3 sentences what the tool results below show for the user's request. "
                  "Use ONLY values present in the results. If something failed, say it could not be verified.\n"
                  f"Request: {message}\nResults: {blob}")
        try:
            out = self.llm.chat([{"role": "system", "content": "You summarise tool output faithfully."},
                                 {"role": "user", "content": prompt}], None)
        except LLMError as exc:
            log_event(self.settings.log_file, "llm_error", request_id=request_id, error=str(exc))
            return None
        return out.content.strip() or None

    # ------------------------------------------------------------------ entry point
    def handle(self, message: str) -> AgentResponse:
        request_id = uuid.uuid4().hex[:12]
        log_event(self.settings.log_file, "request", request_id=request_id, message=message)
        plan = planner.plan(message)
        evidence: list[ToolResult] | None = None
        mode = "workflow"

        mutating = any(self.registry[st.tool].mutating for st in plan.steps if st.tool in self.registry)
        if (plan.multi or mutating) and plan.steps:
            # Multi-tool investigations need ordered, dependency-aware steps; state-changing requests
            # must never take arguments invented by a small model (it once sent namespace='namespace').
            evidence = self._run_plan(plan.steps, request_id)
        else:
            if self.llm is not None:
                evidence = self._run_with_llm(message, request_id)
                mode = "llm-tools"
            # LLM chose nothing, or every call it made was rejected as invalid: the planner takes over.
            llm_useless = not evidence or all(r.status == "invalid" for r in evidence)
            if llm_useless and plan.steps:
                mode = "rules-fallback" if self.llm is not None else "rules"
                rejected = evidence or []
                evidence = self._run_plan(plan.steps, request_id)
                if rejected:
                    log_event(self.settings.log_file, "llm_calls_rejected", request_id=request_id,
                              tools=[r.tool for r in rejected], errors=[r.error for r in rejected])
        evidence = evidence or []

        report = build_report(message, evidence, integration=plan.intent == "integration")
        summary = self._summarise(message, evidence, request_id)
        answer = (f"**Summary (LLM, derived from the evidence below)**\n{summary}\n\n{report}"
                  if summary else report)
        pending = [a for a in self._pending.values() if a.request_id == request_id]
        log_event(self.settings.log_file, "response", request_id=request_id, mode=mode,
                  intent=plan.intent, tools=[r.tool for r in evidence],
                  statuses=[r.status for r in evidence])
        return AgentResponse(request_id=request_id, answer=answer, mode=mode,
                             tool_calls=evidence, pending_actions=pending)
