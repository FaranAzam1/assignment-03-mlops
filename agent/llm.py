"""Thin client for a local Ollama server (chat + native tool calling)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from agent.config import Settings, get_settings


class LLMError(Exception):
    """The language model could not be reached or answered badly."""


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{name, arguments}]


class LLM(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> LLMResponse: ...


class OllamaClient:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    def chat(self, messages, tools=None) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.s.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if tools:
            payload["tools"] = tools
        try:
            resp = requests.post(f"{self.s.ollama_url}/api/chat", json=payload, timeout=self.s.llm_timeout_s)
        except requests.RequestException as exc:
            raise LLMError(f"Ollama unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise LLMError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
        msg = resp.json().get("message", {})
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                import json

                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append({"name": fn.get("name", ""), "arguments": args})
        return LLMResponse(content=msg.get("content", "") or "", tool_calls=calls)
