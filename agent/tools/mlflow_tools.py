"""MLflow tools. Uses MlflowClient only (no web-UI scraping). Read-only."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from agent.config import get_settings
from agent.tools.base import ToolError, params, tool

_client_factory = None  # tests may inject a fake client factory


def set_client_factory(factory) -> None:
    global _client_factory
    _client_factory = factory


def get_client():
    if _client_factory is not None:
        return _client_factory()
    # Fail fast when the server is down instead of retrying for minutes.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "8")
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=get_settings().mlflow_tracking_uri)


def _iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def _run_summary(run) -> dict[str, Any]:
    d = run.data
    return {
        "run_id": run.info.run_id,
        "run_name": d.tags.get("mlflow.runName"),
        "status": run.info.status,
        "started": _iso(run.info.start_time),
        "metrics": dict(d.metrics),
        "params": dict(d.params),
        "dvc_md5": d.tags.get("dvc.md5"),
        "quality_gate": d.tags.get("quality_gate"),
    }


def _experiment_id(client, name_or_id: str) -> str:
    exp = client.get_experiment_by_name(name_or_id)
    if exp is not None:
        return exp.experiment_id
    try:
        return client.get_experiment(name_or_id).experiment_id
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"experiment '{name_or_id}' not found") from exc


def _recent_runs(client, experiment: str, limit: int):
    exp_id = _experiment_id(client, experiment)
    return client.search_runs(
        [exp_id], order_by=["attributes.start_time DESC"], max_results=limit
    )


@tool("mlflow_list_experiments", "mlflow", "List MLflow experiments.")
def mlflow_list_experiments() -> dict[str, Any]:
    exps = get_client().search_experiments()
    return {
        "count": len(exps),
        "experiments": [
            {"experiment_id": e.experiment_id, "name": e.name, "lifecycle": e.lifecycle_stage}
            for e in exps
        ],
    }


_EXP = {"type": "string", "description": "Experiment name or id; defaults to the pipeline's"}


@tool(
    "mlflow_list_runs",
    "mlflow",
    "List the most recent training runs of an experiment with metrics and status.",
    params({"experiment": _EXP, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}),
)
def mlflow_list_runs(experiment: str | None = None, limit: int = 10) -> dict[str, Any]:
    experiment = experiment or get_settings().default_experiment
    runs = _recent_runs(get_client(), experiment, limit)
    return {"experiment": experiment, "count": len(runs), "runs": [_run_summary(r) for r in runs]}


@tool(
    "mlflow_get_run_details",
    "mlflow",
    "Get metrics, parameters, tags and artifacts of one run.",
    params({"run_id": {"type": "string"}}, ["run_id"]),
)
def mlflow_get_run_details(run_id: str) -> dict[str, Any]:
    client = get_client()
    run = client.get_run(run_id)
    artifacts = [a.path for a in client.list_artifacts(run_id)]
    return {
        **_run_summary(run),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
        "artifacts": artifacts,
    }


@tool(
    "mlflow_find_best_run",
    "mlflow",
    "Find the best of the most recent finished runs by a metric. Returns the ranking, "
    "the winner and caveats. A top metric alone does not make a model production-ready.",
    params(
        {
            "experiment": _EXP,
            "metric": {"type": "string", "description": "Metric name, e.g. f1"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "higher_is_better": {"type": "boolean"},
        }
    ),
)
def mlflow_find_best_run(
    experiment: str | None = None,
    metric: str | None = None,
    limit: int = 5,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    s = get_settings()
    experiment, metric = experiment or s.default_experiment, metric or s.primary_metric
    runs = [r for r in _recent_runs(get_client(), experiment, limit) if r.info.status == "FINISHED"]
    scored = [r for r in runs if metric in r.data.metrics]
    if not scored:
        raise ToolError(f"no finished runs with metric '{metric}' in '{experiment}'")
    # Ties on the metric are broken in favour of runs that passed the quality gate.
    sign = 1 if higher_is_better else -1
    scored.sort(key=lambda r: (sign * r.data.metrics[metric], r.data.tags.get("quality_gate") == "passed"),
                reverse=True)
    best = scored[0]
    gate = best.data.tags.get("quality_gate")
    caveats = [
        f"Ranked only the {len(scored)} most recent finished runs, not all history.",
        f"Selection used '{metric}' alone; precision/recall trade-offs and data drift not assessed.",
    ]
    if gate != "passed":
        caveats.append(f"Run's quality_gate tag is '{gate}', so it is NOT cleared for promotion.")
    if len(scored) > 1:
        gap = best.data.metrics[metric] - scored[1].data.metrics[metric]
        if abs(gap) < 0.01:
            caveats.append(f"Margin over the runner-up is only {gap:.4f}: effectively a tie.")
    return {
        "experiment": experiment,
        "metric": metric,
        "compared_runs": len(scored),
        "best": _run_summary(best),
        "ranking": [
            {"run_id": r.info.run_id, metric: r.data.metrics[metric], "started": _iso(r.info.start_time)}
            for r in scored
        ],
        "production_ready": False,
        "caveats": caveats,
    }


@tool(
    "mlflow_compare_runs",
    "mlflow",
    "Compare specific runs side by side (metrics and params).",
    params(
        {"run_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 10}},
        ["run_ids"],
    ),
)
def mlflow_compare_runs(run_ids: list[str]) -> dict[str, Any]:
    client = get_client()
    rows = [_run_summary(client.get_run(r)) for r in run_ids]
    metrics = sorted({m for r in rows for m in r["metrics"]})
    return {"runs": rows, "metric_names": metrics}


@tool(
    "mlflow_compare_experiments",
    "mlflow",
    "Compare two or more experiments by best and mean value of a metric over recent runs.",
    params(
        {
            "experiments": {"type": "array", "items": {"type": "string"}, "minItems": 2},
            "metric": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        },
        ["experiments"],
    ),
)
def mlflow_compare_experiments(
    experiments: list[str], metric: str | None = None, limit: int = 10
) -> dict[str, Any]:
    metric = metric or get_settings().primary_metric
    client = get_client()
    out = []
    for name in experiments:
        vals = [
            r.data.metrics[metric]
            for r in _recent_runs(client, name, limit)
            if metric in r.data.metrics
        ]
        out.append(
            {
                "experiment": name,
                "runs_with_metric": len(vals),
                "best": max(vals) if vals else None,
                "mean": round(sum(vals) / len(vals), 4) if vals else None,
            }
        )
    return {"metric": metric, "experiments": out}


@tool(
    "mlflow_get_model_registry",
    "mlflow",
    "Get registered models and their versions, aliases and source runs (registry information).",
    params({"model_name": {"type": "string"}}),
)
def mlflow_get_model_registry(model_name: str | None = None) -> dict[str, Any]:
    client = get_client()
    models = client.search_registered_models()
    if model_name:
        models = [m for m in models if m.name == model_name]
    result = []
    for m in models:
        versions = client.search_model_versions(f"name='{m.name}'")
        result.append(
            {
                "name": m.name,
                "aliases": dict(getattr(m, "aliases", {}) or {}),
                "versions": [
                    {
                        "version": v.version,
                        "run_id": v.run_id,
                        "status": v.status,
                        "aliases": list(getattr(v, "aliases", []) or []),
                    }
                    for v in versions
                ],
            }
        )
    return {"count": len(result), "models": result}
