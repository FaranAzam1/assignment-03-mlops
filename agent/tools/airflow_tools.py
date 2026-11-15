"""Apache Airflow 3 tools using the authenticated REST API (/auth/token + /api/v2)."""

from __future__ import annotations

import time
from typing import Any

import requests

from agent.config import get_settings
from agent.tools.base import PermissionDenied, ToolError, ToolUnavailable, params, tool

_session_factory = requests.Session  # tests inject a fake
_token: dict[str, Any] = {"value": None, "at": 0.0}
_TOKEN_TTL_S = 600


def set_session_factory(factory) -> None:
    global _session_factory
    _session_factory = factory
    _token.update(value=None, at=0.0)


def _call(method: str, path: str, **kwargs) -> Any:
    s = get_settings()
    http = _session_factory()
    timeout = min(s.tool_timeout_s, 15)
    if not s.airflow_password:
        raise ToolUnavailable("AIRFLOW_PASSWORD is not configured")
    if _token["value"] is None or time.time() - _token["at"] > _TOKEN_TTL_S:
        resp = http.post(
            f"{s.airflow_url}/auth/token",
            json={"username": s.airflow_username, "password": s.airflow_password},
            timeout=timeout,
        )
        if resp.status_code in (401, 403):
            raise ToolError("Airflow rejected the configured credentials")
        if resp.status_code >= 400:
            raise ToolError(f"Airflow token request failed: HTTP {resp.status_code}")
        _token.update(value=resp.json()["access_token"], at=time.time())
    headers = {"Authorization": f"Bearer {_token['value']}", "Accept": "application/json"}
    resp = http.request(method, f"{s.airflow_url}/api/v2{path}", headers=headers, timeout=timeout, **kwargs)
    if resp.status_code == 401:
        _token.update(value=None, at=0.0)
        raise ToolError("Airflow token expired or invalid")
    if resp.status_code == 404:
        raise ToolError(f"Airflow resource not found: {path}")
    if resp.status_code >= 400:
        raise ToolError(f"Airflow API error HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()


def _check_dag(dag_id: str) -> None:
    allowed = get_settings().allowed_dag_ids
    if dag_id not in allowed:
        raise PermissionDenied(f"DAG '{dag_id}' is not in the approved list {list(allowed)}")


def _run_view(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "dag_run_id": r.get("dag_run_id"),
        "state": r.get("state"),
        "run_type": r.get("run_type"),
        "start_date": r.get("start_date"),
        "end_date": r.get("end_date"),
        "conf": r.get("conf"),
    }


_DAG = {"type": "string", "description": "DAG id, e.g. training_pipeline"}


@tool("airflow_list_dags", "airflow", "List DAGs known to Airflow with paused state.")
def airflow_list_dags() -> dict[str, Any]:
    data = _call("GET", "/dags", params={"limit": 100})
    dags = [
        {
            "dag_id": d["dag_id"],
            "is_paused": d.get("is_paused"),
            "schedule": d.get("timetable_summary"),
            "tags": [t["name"] for t in d.get("tags", [])],
            "approved_for_agent": d["dag_id"] in get_settings().allowed_dag_ids,
        }
        for d in data.get("dags", [])
    ]
    return {"count": len(dags), "dags": dags}


@tool(
    "airflow_get_dag_runs",
    "airflow",
    "Get the most recent runs of a DAG and their states (success/failed/running).",
    params({"dag_id": _DAG, "limit": {"type": "integer", "minimum": 1, "maximum": 25}}),
)
def airflow_get_dag_runs(dag_id: str | None = None, limit: int = 5) -> dict[str, Any]:
    dag_id = dag_id or get_settings().allowed_dag_ids[0]
    data = _call("GET", f"/dags/{dag_id}/dagRuns", params={"limit": limit, "order_by": "-start_date"})
    runs = [_run_view(r) for r in data.get("dag_runs", [])]
    return {"dag_id": dag_id, "count": len(runs), "runs": runs}


def _latest_run_id(dag_id: str, only_failed: bool = False) -> str:
    runs = airflow_get_dag_runs(dag_id, 25)["runs"]
    if only_failed:
        runs = [r for r in runs if r["state"] == "failed"]
    if not runs:
        raise ToolError(f"no {'failed ' if only_failed else ''}runs found for DAG '{dag_id}'")
    return runs[0]["dag_run_id"]


@tool(
    "airflow_get_failed_tasks",
    "airflow",
    "Identify failed (or upstream-failed) tasks in a DAG run; defaults to the latest failed run.",
    params({"dag_id": _DAG, "dag_run_id": {"type": "string"}}),
)
def airflow_get_failed_tasks(dag_id: str | None = None, dag_run_id: str | None = None) -> dict[str, Any]:
    dag_id = dag_id or get_settings().allowed_dag_ids[0]
    dag_run_id = dag_run_id or _latest_run_id(dag_id, only_failed=True)
    data = _call("GET", f"/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances")
    tasks = data.get("task_instances", [])
    failed = [
        {
            "task_id": t["task_id"],
            "state": t["state"],
            "try_number": t.get("try_number"),
            "max_tries": t.get("max_tries"),
        }
        for t in tasks
        if t.get("state") in ("failed", "upstream_failed")
    ]
    return {
        "dag_id": dag_id,
        "dag_run_id": dag_run_id,
        "failed_tasks": failed,
        "all_task_states": {t["task_id"]: t["state"] for t in tasks},
    }


@tool(
    "airflow_get_task_diagnostics",
    "airflow",
    "Get execution info and the log tail of one task instance for failure diagnosis.",
    params(
        {
            "dag_id": _DAG,
            "dag_run_id": {"type": "string", "description": "Defaults to latest failed run"},
            "task_id": {"type": "string"},
        },
        ["task_id"],
    ),
)
def airflow_get_task_diagnostics(
    task_id: str, dag_id: str | None = None, dag_run_id: str | None = None
) -> dict[str, Any]:
    dag_id = dag_id or get_settings().allowed_dag_ids[0]
    dag_run_id = dag_run_id or _latest_run_id(dag_id, only_failed=True)
    base = f"/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}"
    ti = _call("GET", base)
    try_number = max(int(ti.get("try_number") or 1), 1)
    log_lines: list[str] = []
    errors: list[str] = []
    try:
        log = _call("GET", f"{base}/logs/{try_number}")
        for item in log.get("content", []):
            if isinstance(item, str):
                line, is_err = item, any(k in item.lower() for k in ("error", "exception", "traceback"))
            else:
                event = str(item.get("event", ""))
                if event.startswith("::"):  # group markers carry no information
                    continue
                line = f"{str(item.get('level', 'info')).upper()} {event}"
                is_err = item.get("level") in ("error", "critical") or bool(item.get("error_detail"))
                for exc in item.get("error_detail") or []:
                    detail = f"{exc.get('exc_type')}: {exc.get('exc_value')}"
                    errors.append(detail)
                    line += f" -> {detail}"
            log_lines.append(line)
            if is_err and line not in errors:
                errors.append(line)
    except ToolError as exc:
        log_lines.append(f"[log unavailable: {exc}]")
    return {
        "dag_id": dag_id,
        "dag_run_id": dag_run_id,
        "task_id": task_id,
        "state": ti.get("state"),
        "try_number": ti.get("try_number"),
        "max_tries": ti.get("max_tries"),
        "duration_s": ti.get("duration"),
        "start_date": ti.get("start_date"),
        "end_date": ti.get("end_date"),
        "error_lines": errors[-8:],
        "log_tail": log_lines[-25:],
    }


@tool(
    "airflow_trigger_dag",
    "airflow",
    "Trigger a run of an approved training DAG. Changes state: requires explicit user confirmation.",
    params(
        {
            "dag_id": _DAG,
            "conf": {"type": "object", "description": "Optional run parameters"},
        },
        ["dag_id"],
    ),
    mutating=True,
)
def airflow_trigger_dag(dag_id: str, conf: dict | None = None, _confirmed: bool = False) -> dict[str, Any]:
    _check_dag(dag_id)
    if not _confirmed:
        raise PermissionDenied("triggering a DAG requires explicit user confirmation")
    run = _call("POST", f"/dags/{dag_id}/dagRuns", json={"logical_date": None, "conf": conf or {}})
    return {"triggered": True, **_run_view(run)}
