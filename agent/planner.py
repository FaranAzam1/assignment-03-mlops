"""Deterministic request classification and tool workflows.

Used (a) for multi-technology requests, where ordering and data dependencies between tools
matter (the best run's id feeds the DVC provenance lookup) and (b) as a fallback when the LLM
is unavailable or selects no tool. Operational facts always come from tool calls.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.schemas import ToolResult

Evidence = list[ToolResult]
ArgsFn = Callable[[Evidence], dict[str, Any] | None]


@dataclass
class Step:
    tool: str
    args: dict[str, Any] | ArgsFn = field(default_factory=dict)

    def resolve(self, evidence: Evidence) -> dict[str, Any] | None:
        return self.args(evidence) if callable(self.args) else dict(self.args)


MODULE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "dvc": ("dvc", "dataset", "data version", "provenance", "data changed", "reproduce"),
    "mlflow": ("mlflow", "experiment", "metric", "f1", "accuracy", "best model", "best run",
               "registry", "training run", "recent model", "best recent"),
    "airflow": ("airflow", "dag", "pipeline", "workflow", "retrain", "task"),
    "kubernetes": ("kubernetes", "k8s", "pod", "deployment", "deploy", "service", "replica",
                   "prediction api", "unavailable", "namespace", "logs", "cluster"),
}
RUN_ID_RE = re.compile(r"\b[0-9a-f]{32}\b")
TAG_RE = re.compile(r"\bdata-v\d+\b|\bv\d+(?:\.\d+)?\b")
METRICS = ("f1", "accuracy", "precision", "recall", "roc_auc")
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10}


def modules_in(message: str) -> list[str]:
    low = message.lower()
    return [m for m, kws in MODULE_KEYWORDS.items() if any(k in low for k in kws)]


def _last_n(low: str, default: int) -> int:
    m = re.search(r"(?:last|recent|latest|past)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")", low)
    if not m:
        return default
    tok = m.group(1)
    return int(tok) if tok.isdigit() else NUMBER_WORDS[tok]


def _metric(low: str) -> str:
    return next((m for m in METRICS if m in low.replace("-", "_").replace(" ", "_")), "f1")


def _tag(low: str, index: int, default: str) -> str:
    tags = [t if t.startswith("data-") else f"data-{t}" for t in TAG_RE.findall(low)]
    return tags[index] if len(tags) > index else default


def _result(evidence: Evidence, tool: str) -> ToolResult | None:
    for r in reversed(evidence):
        if r.tool == tool and r.ok:
            return r
    return None


def _best_run_id(evidence: Evidence) -> dict[str, Any] | None:
    r = _result(evidence, "mlflow_find_best_run")
    return {"run_id": r.data["best"]["run_id"]} if r else None


def _first_unhealthy_pod(evidence: Evidence) -> dict[str, Any] | None:
    r = _result(evidence, "k8s_find_unhealthy_pods")
    if r and r.data["unhealthy_pods"]:
        return {"pod": r.data["unhealthy_pods"][0]["name"], "previous": False}
    return None


def _first_failed_task(evidence: Evidence) -> dict[str, Any] | None:
    r = _result(evidence, "airflow_get_failed_tasks")
    if r and r.data["failed_tasks"]:
        failed = [t for t in r.data["failed_tasks"] if t["state"] == "failed"] or r.data["failed_tasks"]
        return {"task_id": failed[0]["task_id"], "dag_run_id": r.data["dag_run_id"]}
    return None


def _failed_run_args(evidence: Evidence) -> dict[str, Any] | None:
    r = _result(evidence, "airflow_get_dag_runs")
    if not r:
        return None
    failed = [x for x in r.data["runs"] if x["state"] == "failed"]
    return {"dag_id": r.data["dag_id"], "dag_run_id": failed[0]["dag_run_id"]} if failed else None


def _mlflow_steps(low: str, msg: str) -> list[Step]:
    run_id = RUN_ID_RE.search(msg)
    if "registry" in low or "registered" in low:
        return [Step("mlflow_get_model_registry")]
    if run_id and "dvc" not in low and "dataset" not in low:
        return [Step("mlflow_get_run_details", {"run_id": run_id.group(0)})]
    if "experiments" in low and not any(k in low for k in ("best", "compare", "recommend")):
        return [Step("mlflow_list_experiments")]
    n = _last_n(low, 5)
    metric = _metric(low)
    if "compare" in low or "recommend" in low:
        return [Step("mlflow_list_runs", {"limit": n}),
                Step("mlflow_find_best_run", {"metric": metric, "limit": n})]
    if any(k in low for k in ("list runs", "recent runs", "show runs", "training runs")) and "best" not in low:
        return [Step("mlflow_list_runs", {"limit": n})]
    return [Step("mlflow_find_best_run", {"metric": metric, "limit": n})]


def _dvc_steps(low: str, msg: str, with_best_run: bool) -> list[Step]:
    run_id = RUN_ID_RE.search(msg)
    if run_id:
        return [Step("dvc_get_run_dataset_provenance", {"run_id": run_id.group(0)})]
    if with_best_run:
        return [Step("dvc_get_run_dataset_provenance", _best_run_id), Step("dvc_get_dataset_version")]
    if any(k in low for k in ("reproduce", "restore", "checkout", "go back")):
        return [Step("dvc_explain_reproduce", {"rev": _tag(low, 0, "data-v1")})]
    if "compare" in low or "difference" in low or " diff" in low:
        return [Step("dvc_compare_dataset_versions", {"rev_a": _tag(low, 0, "data-v1"), "rev_b": _tag(low, 1, "HEAD")})]
    if any(k in low for k in ("changed", "modified", "drift", "up to date", "untracked")):
        return [Step("dvc_check_data_changed")]
    if any(k in low for k in ("history", "versions")):
        return [Step("dvc_list_dataset_history")]
    if any(k in low for k in ("list", "tracked", "which datasets")):
        return [Step("dvc_list_datasets")]
    return [Step("dvc_get_dataset_version")]


def _airflow_steps(low: str, with_gate: bool) -> list[Step]:
    if any(k in low for k in ("trigger", "start the", "kick off", "run the model training", "run the training")):
        return [Step("airflow_trigger_dag", {"dag_id": "training_pipeline"})]
    if any(k in low for k in ("list dags", "which dags", "available dags")):
        return [Step("airflow_list_dags")]
    steps = [Step("airflow_get_dag_runs", {"dag_id": "training_pipeline", "limit": 5})]
    if with_gate:  # only drill into a run if one of the recent runs actually failed
        steps += [Step("airflow_get_failed_tasks", _failed_run_args),
                  Step("airflow_get_task_diagnostics", lambda ev: (
                      {**t, "dag_id": "training_pipeline"} if (t := _first_failed_task(ev)) else None))]
    elif any(k in low for k in ("fail", "error", "why", "diagnos", "broke")):
        steps += [Step("airflow_get_failed_tasks", {"dag_id": "training_pipeline"}),
                  Step("airflow_get_task_diagnostics", lambda ev: (
                      {**t, "dag_id": "training_pipeline"} if (t := _first_failed_task(ev)) else None))]
    return steps


def _k8s_steps(low: str, msg: str) -> list[Step]:
    pod = re.search(r"\b(ml-inference-[a-z0-9-]+)\b", low)
    if "restart" in low:
        return [Step("k8s_restart_deployment", {"deployment": "ml-inference"})]
    if "scale" in low:
        n = re.search(r"(?:to|replicas?)\s*(\d+)|(\d+)\s*replicas?", low)
        count = int(next(g for g in n.groups() if g)) if n else 2
        return [Step("k8s_scale_deployment", {"deployment": "ml-inference", "replicas": count})]
    if "log" in low:
        if pod:
            return [Step("k8s_get_pod_logs", {"pod": pod.group(1)})]
        return [Step("k8s_find_unhealthy_pods"), Step("k8s_list_pods"),
                Step("k8s_get_pod_logs", lambda ev: _first_unhealthy_pod(ev) or (
                    {"pod": r.data["pods"][0]["name"]} if (r := _result(ev, "k8s_list_pods")) and r.data["pods"] else None))]
    if any(k in low for k in ("unavailable", "down", "not working", "unhealthy", "crash", "fail",
                              "error", "why", "diagnos", "broken", "not ready", "restart")):
        return [Step("k8s_find_unhealthy_pods"), Step("k8s_list_deployments"), Step("k8s_list_services"),
                Step("k8s_get_pod_logs", _first_unhealthy_pod),
                Step("k8s_get_deployed_version")]
    if "service" in low:
        return [Step("k8s_list_services")]
    if "version" in low:
        return [Step("k8s_get_deployed_version")]
    if "deployment" in low:
        return [Step("k8s_list_deployments")]
    if "pod" in low:
        return [Step("k8s_list_pods")]
    return [Step("k8s_get_deployed_version"), Step("k8s_find_unhealthy_pods")]


@dataclass
class Plan:
    intent: str
    modules: list[str]
    steps: list[Step]

    @property
    def multi(self) -> bool:
        return len(self.modules) >= 2


def plan(message: str) -> Plan:
    low = message.lower()
    modules = modules_in(message)
    if "best" in low and "model" in low and any(m in modules for m in ("kubernetes", "dvc", "airflow")):
        modules = sorted(set(modules) | {"mlflow"})
    steps: list[Step] = []
    if len(modules) >= 3 or ("mlflow" in modules and "dvc" in modules and "kubernetes" in modules):
        # Mandatory integration scenario: model -> data provenance -> deployment -> pipeline.
        steps = (_mlflow_steps("best", message)[-1:] + _dvc_steps(low, message, True)
                 + [Step("k8s_get_deployed_version"), Step("k8s_find_unhealthy_pods")]
                 + _airflow_steps("status", with_gate=True))
        # recent runs, so failed quality gates are visible
        return Plan("integration", ["mlflow", "dvc", "kubernetes", "airflow"], steps)
    order = ["mlflow", "dvc", "kubernetes", "airflow"]
    with_best = "mlflow" in modules and "dvc" in modules
    for m in order:
        if m not in modules:
            continue
        if m == "mlflow":
            steps += _mlflow_steps(low, message)
        elif m == "dvc":
            steps += _dvc_steps(low, message, with_best)
        elif m == "kubernetes":
            steps += _k8s_steps(low, message)
        else:
            steps += _airflow_steps(low, with_gate=False)
    intent = "+".join(modules) if modules else "unknown"
    return Plan(intent, modules, steps)
