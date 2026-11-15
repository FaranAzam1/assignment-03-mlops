"""Tool runner: validation, timeouts, error mapping, structured logs without secrets."""

import time

from agent.logging_utils import redact
from agent.tools.base import (
    PermissionDenied,
    Tool,
    ToolUnavailable,
    execute_tool,
    params,
    run_command,
)
from tests.conftest import read_log


def _reg(**funcs):
    return {name: Tool(name, "dvc", "test tool", schema, fn) for name, (fn, schema) in funcs.items()}


def test_successful_execution_and_log_fields(settings):
    reg = _reg(echo=(lambda text: {"echo": text}, params({"text": {"type": "string"}}, ["text"])))
    res = execute_tool("echo", {"text": "hi"}, request_id="req123", registry=reg)
    assert res.ok and res.data == {"echo": "hi"} and res.request_id == "req123"
    rec = read_log(settings)[-1]
    assert rec["tool"] == "echo" and rec["status"] == "ok"
    assert rec["request_id"] == "req123" and rec["duration_ms"] >= 0


def test_invalid_arguments_rejected_before_execution():
    called = []
    reg = _reg(t=(lambda n: called.append(n), params({"n": {"type": "integer"}}, ["n"])))
    res = execute_tool("t", {"n": "not-an-int"}, registry=reg)
    assert res.status == "invalid" and not called
    assert execute_tool("t", {}, registry=reg).status == "invalid"
    assert execute_tool("t", {"n": 1, "extra": 1}, registry=reg).status == "invalid"


def test_unknown_tool_is_invalid():
    assert execute_tool("does_not_exist", {}, registry={}).status == "invalid"


def test_tool_timeout():
    reg = _reg(slow=(lambda: time.sleep(5), params()))
    res = execute_tool("slow", {}, registry=reg)
    assert res.status == "timeout" and "timed out" in res.error
    assert res.data is None


def test_service_failure_statuses():
    def refuse():
        raise ConnectionError("Connection refused")

    def deny():
        raise PermissionDenied("nope")

    def down():
        raise ToolUnavailable("gone")

    def bug():
        raise ValueError("bad data")

    reg = _reg(a=(refuse, params()), b=(deny, params()), c=(down, params()), d=(bug, params()))
    assert execute_tool("a", {}, registry=reg).status == "unavailable"
    assert execute_tool("b", {}, registry=reg).status == "denied"
    assert execute_tool("c", {}, registry=reg).status == "unavailable"
    assert execute_tool("d", {}, registry=reg).status == "error"


def test_logs_never_contain_secrets(settings):
    reg = _reg(
        login=(
            lambda password, note: "ok",
            params({"password": {"type": "string"}, "note": {"type": "string"}}),
        )
    )
    execute_tool("login", {"password": "hunter2", "note": "Bearer abc.def.ghi"}, registry=reg)
    raw = settings.log_file.read_text()
    assert "hunter2" not in raw and "abc.def.ghi" not in raw and "REDACTED" in raw


def test_redact_nested_and_truncates():
    out = redact({"a": {"api_key": "k", "x": "y" * 1000}, "token": "t"})
    assert out["a"]["api_key"] == out["token"] == "***REDACTED***"
    assert out["a"]["x"].endswith("[truncated]")


def test_subprocess_allowlist(tmp_path):
    for bad in (["git", "push"], ["git", "reset", "--hard"], ["rm", "-rf", "x"], ["dvc", "push"]):
        try:
            run_command(bad, tmp_path)
        except PermissionDenied:
            continue
        raise AssertionError(f"{bad} should have been denied")


def test_read_only_tool_retries_once_on_transient_connection_error():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("Connection aborted")
        return "recovered"

    res = execute_tool("flaky", {}, registry=_reg(flaky=(flaky, params())))
    assert res.ok and res.data == "recovered" and len(calls) == 2


def test_mutating_tool_is_never_retried():
    calls = []

    def boom(_confirmed=False):
        calls.append(1)
        raise ConnectionError("Connection aborted")

    reg = {"boom": Tool("boom", "kubernetes", "x", params(), boom, mutating=True)}
    res = execute_tool("boom", {}, registry=reg, confirmed=True)
    assert res.status == "unavailable" and len(calls) == 1
