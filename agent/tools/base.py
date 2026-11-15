"""Tool abstraction: registry, argument validation, timeouts, error mapping, logging."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from agent.config import Settings, get_settings
from agent.logging_utils import log_event
from agent.schemas import ToolResult, ToolSpec


class ToolError(Exception):
    """Generic tool failure (service answered with an error, bad data, ...)."""


class ToolUnavailable(ToolError):
    """The backing service or binary cannot be reached."""


class PermissionDenied(ToolError):
    """The operation is not permitted (no confirmation, namespace or DAG not approved)."""


@dataclass(frozen=True)
class Tool:
    name: str
    module: str  # dvc | mlflow | airflow | kubernetes
    description: str
    parameters: dict[str, Any]
    func: Callable[..., Any]
    mutating: bool = False
    confirm_param: bool = field(default=False)

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            module=self.module,
            description=self.description,
            mutating=self.mutating,
            parameters=self.parameters,
        )

    def llm_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def params(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    _REGISTRY[tool.name] = tool
    return tool


def tool(
    name: str,
    module: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    mutating: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a function as an agent tool."""

    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        register(
            Tool(
                name=name,
                module=module,
                description=description,
                parameters=parameters or params(),
                func=func,
                mutating=mutating,
            )
        )
        return func

    return wrap


def get_registry() -> dict[str, Tool]:
    # Importing the modules registers their tools.
    from agent.tools import airflow_tools, dvc_tools, kubernetes_tools, mlflow_tools  # noqa: F401

    return _REGISTRY


def _is_connection_error(exc: BaseException) -> bool:
    low = f"{type(exc).__name__}: {exc}".lower()
    return any(s in low for s in ("connection", "max retries", "refused", "timed out", "unreachable"))


_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool")


def execute_tool(
    name: str,
    args: dict[str, Any] | None,
    request_id: str | None = None,
    confirmed: bool = False,
    settings: Settings | None = None,
    registry: dict[str, Tool] | None = None,
) -> ToolResult:
    """Run a registered tool and always return a ToolResult (never raises).

    `confirmed` is set only by the orchestrator after an explicit user confirmation. It is
    never taken from the LLM's arguments.
    """
    settings = settings or get_settings()
    registry = registry if registry is not None else get_registry()
    request_id = request_id or uuid.uuid4().hex[:12]
    args = dict(args or {})
    started = time.perf_counter()
    status, data, error = "ok", None, None

    t = registry.get(name)
    if t is None:
        status, error = "invalid", f"unknown tool '{name}'"
    else:
        try:
            jsonschema.validate(args, t.parameters)
        except jsonschema.ValidationError as exc:
            status, error = "invalid", f"invalid arguments for {name}: {exc.message}"
        else:
            if t.mutating:
                args["_confirmed"] = bool(confirmed)
            # Read-only tools get one retry on a connection-level failure (busy server, dropped socket).
            attempts = 1 if t.mutating else 2
            future = _POOL.submit(t.func, **args)
            try:
                for attempt in range(attempts):
                    try:
                        data = future.result(timeout=settings.tool_timeout_s)
                        break
                    except (FutureTimeout, ToolError, PermissionDenied):
                        raise
                    except Exception as exc:  # noqa: BLE001
                        if attempt + 1 >= attempts or not _is_connection_error(exc):
                            raise
                        future = _POOL.submit(t.func, **args)
            except FutureTimeout:
                future.cancel()
                status = "timeout"
                error = f"{name} timed out after {settings.tool_timeout_s:.0f}s"
            except PermissionDenied as exc:
                status, error = "denied", str(exc)
            except ToolUnavailable as exc:
                status, error = "unavailable", str(exc)
            except ToolError as exc:
                status, error = "error", str(exc)
            except Exception as exc:  # noqa: BLE001 - tools wrap arbitrary client libraries
                # Connection problems from requests/urllib3/kubernetes/mlflow land here.
                text = f"{type(exc).__name__}: {exc}"
                status = "unavailable" if _is_connection_error(exc) else "error"
                error = text[:400]
            args.pop("_confirmed", None)

    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    result = ToolResult(
        tool=name,
        status=status,
        data=data if status == "ok" else None,
        error=error,
        duration_ms=duration_ms,
        request_id=request_id,
        args=args,
    )
    log_event(
        settings.log_file,
        "tool_call",
        request_id=request_id,
        tool=name,
        status=status,
        duration_ms=duration_ms,
        args=args,
        error=error,
    )
    return result


# ---------------------------------------------------------------- subprocess helper

ALLOWED_COMMANDS: dict[str, set[str]] = {
    "git": {"log", "show", "rev-parse", "tag", "status", "diff"},
    "dvc": {"status", "diff", "list", "remote", "data", "config"},
}


def find_binary(name: str) -> str | None:
    scripts = Path(sys.executable).parent
    for cand in (scripts / f"{name}.exe", scripts / name):
        if cand.exists():
            return str(cand)
    return shutil.which(name)


def run_command(argv: list[str], cwd: Path, timeout: float = 15.0) -> str:
    """Run an allow-listed read-only command. Raises ToolUnavailable / PermissionDenied."""
    if len(argv) < 2 or argv[0] not in ALLOWED_COMMANDS:
        raise PermissionDenied(f"command not allowed: {' '.join(argv[:2])}")
    if argv[1] not in ALLOWED_COMMANDS[argv[0]]:
        raise PermissionDenied(f"subcommand not allowed: {argv[0]} {argv[1]}")
    binary = find_binary(argv[0])
    if binary is None:
        raise ToolUnavailable(f"'{argv[0]}' executable not found")
    try:
        proc = subprocess.run(  # noqa: S603 - argv is allow-listed above, shell=False
            [binary, *argv[1:]],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolUnavailable(f"{argv[0]} {argv[1]} timed out") from exc
    if proc.returncode != 0:
        raise ToolError(f"{argv[0]} {argv[1]} failed: {(proc.stderr or proc.stdout).strip()[:300]}")
    return proc.stdout
