"""Builds the final answer strictly from tool evidence: facts, uncertainties, actions."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from agent.schemas import ToolResult

DEGRADATION_WORDS = ("degrad", "drift", "dropped", "worse", "accuracy fell", "performance fell")


def _by_tool(evidence: list[ToolResult]) -> dict[str, ToolResult]:
    out: dict[str, ToolResult] = {}
    for r in evidence:
        if r.ok:
            out[r.tool] = r
    return out


# ------------------------------------------------------------------ per-tool fact lines

def _f_dvc_version(d):
    c = d.get("last_changed_in") or {}
    tags = ", ".join(c.get("tags") or []) or "no tag"
    return [f"Dataset {d['path']}: md5 {d['md5']}, last changed in commit {c.get('short')} ({tags}); "
            f"repo HEAD is {d['git_head']}."]


def _f_provenance(d):
    c = d.get("resolved_commit") or {}
    ver = ", ".join(d.get("dataset_version") or []) or "untagged"
    line = (f"Run {d['run_id'][:8]} recorded dataset md5 {d['recorded_md5']} "
            f"(rows: {d.get('rows')}); that data version is Git commit {c.get('short', 'UNKNOWN')} / {ver}.")
    return [line, f"To reproduce: {d['reproduce_hint']}"]


def _f_changed(d):
    return [f"{d['path']}: {d['summary']}."]


def _f_compare(d):
    a, b = d["rev_a"], d["rev_b"]
    same = "identical" if d["identical"] else f"different (size delta {d['size_delta_bytes']} bytes)"
    return [f"{a['rev']} (md5 {a['md5']}) vs {b['rev']} (md5 {b['md5']}): {same}."]


def _f_history(d):
    return [f"{v['commit']} {','.join(v['tags']) or '-'} md5 {v['md5']} - {v['subject']}" for v in d["versions"]]


def _f_reproduce(d):
    return [f"Expected md5 at {d['rev']}: {d['expected_md5']}."] + [f"  $ {s}" for s in d["steps"]]


def _f_list_datasets(d):
    return [f"{x['path']} (md5 {x['md5']}, {x['size']} bytes) via {x['dvc_file']}" for x in d["datasets"]] or [
        "No DVC-tracked datasets found."]


def _metric_line(m: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in sorted(m.items()))


def _f_best(d):
    b = d["best"]
    lines = [f"Best of {d['compared_runs']} recent finished runs by {d['metric']}: run {b['run_id'][:8]} "
             f"({_metric_line(b['metrics'])}); quality_gate tag: {b.get('quality_gate')}."]
    lines += ["Ranking: " + ", ".join(f"{r['run_id'][:8]}={r[d['metric']]:.4f}" for r in d["ranking"])]
    return lines


def _f_runs(d):
    return [f"{r['run_id'][:8]} {r['status']} {r['started']} {_metric_line(r['metrics'])} "
            f"gate={r.get('quality_gate')}" for r in d["runs"]] or ["No runs found."]


def _f_run_details(d):
    return [f"Run {d['run_id'][:8]} {d['status']}: {_metric_line(d['metrics'])}; params {d['params']}; "
            f"artifacts {d['artifacts']}."]


def _f_experiments(d):
    return [f"{e['experiment_id']}: {e['name']}" for e in d["experiments"]]


def _f_compare_exp(d):
    return [f"{e['experiment']}: best {d['metric']}={e['best']}, mean={e['mean']} ({e['runs_with_metric']} runs)"
            for e in d["experiments"]]


def _f_registry(d):
    if not d["models"]:
        return ["No registered models."]
    return [f"Model {m['name']}: " + "; ".join(
        f"v{v['version']} aliases={v['aliases']} run={str(v['run_id'])[:8]}" for v in m["versions"]) for m in d["models"]]


def _f_dag_runs(d):
    return [f"{d['dag_id']} run {r['dag_run_id']}: {r['state']} (start {r['start_date']})" for r in d["runs"]] or [
        f"No runs of {d['dag_id']}."]


def _f_failed_tasks(d):
    ft = ", ".join(f"{t['task_id']}={t['state']}" for t in d["failed_tasks"]) or "none"
    return [f"Run {d['dag_run_id']} failed tasks: {ft}.", f"Task states: {d['all_task_states']}"]


def _f_task_diag(d):
    lines = [f"Task {d['task_id']} state={d['state']} try {d['try_number']}/{d['max_tries']}."]
    lines += [f"  log: {ln[:200]}" for ln in (d["error_lines"] or d["log_tail"][-4:])]
    return lines


def _f_trigger(d):
    return [f"Triggered DAG run {d['dag_run_id']} (state {d['state']})."]


def _f_list_dags(d):
    return [f"{x['dag_id']} paused={x['is_paused']} approved={x['approved_for_agent']}" for x in d["dags"]]


def _f_version(d):
    return [f"Deployment {d['deployment']}: image(s) {d['images']}, ready {d['ready_replicas']}/{d['desired_replicas']}, "
            f"rollout complete: {d['rollout_complete']}, MODEL_VERSION={d['model_version_configmap']}, "
            f"version label={d['version_label']}."]


def _f_unhealthy(d):
    if not d["unhealthy_pods"]:
        return ["All Pods in the namespace look healthy."]
    lines = [f"Unhealthy Pod {p['name']}: {p['problem']} (phase {p['phase']}, restarts {p['restarts']})."
             for p in d["unhealthy_pods"]]
    lines += [f"Event {e['reason']} on {e['object']}: {e['message'][:160]}" for e in d["warning_events"][:4]]
    return lines


def _f_pods(d):
    return [f"{p['name']} phase={p['phase']} ready={p['ready']} restarts={p['restarts']}" for p in d["pods"]]


def _f_deployments(d):
    return [f"{x['name']}: {x['ready']}/{x['desired']} ready, images {x['images']}" for x in d["deployments"]]


def _f_services(d):
    return [f"{x['name']} {x['type']} {x['cluster_ip']} ports {x['ports']} selector {x['selector']}"
            for x in d["services"]]


def _f_logs(d):
    tail = d["lines"][-8:]
    return [f"Logs of {d['pod']}:"] + [f"  {ln[:200]}" for ln in tail] if tail else [
        f"Pod {d['pod']} has no logs ({d.get('note', 'empty')})."]


def _f_events(d):
    return [f"{e['type']} {e['reason']} {e['object']}: {e['message'][:160]}" for e in d["events"][:8]]


def _f_restart(d):
    return [f"Restarted deployment {d['restarted']} in {d['namespace']}."]


def _f_scale(d):
    return [f"Scaled {d['scaled']} to {d['replicas']} replicas in {d['namespace']}."]


FORMATTERS: dict[str, Callable[[Any], list[str]]] = {
    "dvc_get_dataset_version": _f_dvc_version, "dvc_get_run_dataset_provenance": _f_provenance,
    "dvc_check_data_changed": _f_changed, "dvc_compare_dataset_versions": _f_compare,
    "dvc_list_dataset_history": _f_history, "dvc_explain_reproduce": _f_reproduce,
    "dvc_list_datasets": _f_list_datasets, "mlflow_find_best_run": _f_best,
    "mlflow_list_runs": _f_runs, "mlflow_get_run_details": _f_run_details,
    "mlflow_list_experiments": _f_experiments, "mlflow_compare_experiments": _f_compare_exp,
    "mlflow_compare_runs": _f_runs, "mlflow_get_model_registry": _f_registry,
    "airflow_get_dag_runs": _f_dag_runs, "airflow_get_failed_tasks": _f_failed_tasks,
    "airflow_get_task_diagnostics": _f_task_diag, "airflow_trigger_dag": _f_trigger,
    "airflow_list_dags": _f_list_dags, "k8s_get_deployed_version": _f_version,
    "k8s_find_unhealthy_pods": _f_unhealthy, "k8s_list_pods": _f_pods,
    "k8s_list_deployments": _f_deployments, "k8s_list_services": _f_services,
    "k8s_get_pod_logs": _f_logs, "k8s_get_events": _f_events,
    "k8s_restart_deployment": _f_restart, "k8s_scale_deployment": _f_scale,
}


def facts_for(r: ToolResult) -> list[str]:
    fmt = FORMATTERS.get(r.tool)
    try:
        return fmt(r.data) if fmt else [json.dumps(r.data, default=str)[:400]]
    except (KeyError, TypeError, IndexError):  # shape drift must never produce invented text
        return [json.dumps(r.data, default=str)[:400]]


# ------------------------------------------------------------------ retraining decision

def retraining_assessment(evidence: list[ToolResult], message: str) -> dict[str, Any]:
    """Recommend retraining only on evidence. Never triggers anything."""
    got = _by_tool(evidence)
    reasons: list[str] = []
    notes: list[str] = ["No model-freshness policy is configured, so model age alone is not evidence."]

    best = got.get("mlflow_find_best_run")
    if best and best.data["best"].get("quality_gate") != "passed":
        reasons.append(f"Best candidate's quality_gate tag is '{best.data['best'].get('quality_gate')}' (never cleared).")
    elif not best:
        notes.append("Model quality could not be checked (no MLflow evidence).")

    runs = got.get("airflow_get_dag_runs")
    failed = got.get("airflow_get_failed_tasks")
    if failed and any("quality_gate" in t["task_id"] or "gate" in t["task_id"] for t in failed.data["failed_tasks"]):
        reasons.append(f"Latest failed pipeline run {failed.data['dag_run_id']} failed the model quality gate.")
    elif runs and runs.data["runs"] and runs.data["runs"][0]["state"] == "failed":
        notes.append("Latest pipeline run failed but not at the quality gate; fix the failure before judging the model.")
    elif not runs:
        notes.append("Pipeline status could not be checked (no Airflow evidence).")

    prov, cur = got.get("dvc_get_run_dataset_provenance"), got.get("dvc_get_dataset_version")
    if prov and cur and prov.data["recorded_md5"] != cur.data["md5"]:
        reasons.append("The model was trained on a different dataset version than the current committed data "
                       f"({prov.data['recorded_md5'][:8]} vs {cur.data['md5'][:8]}).")

    if any(w in message.lower() for w in DEGRADATION_WORDS):
        reasons.append("You reported model degradation (unverified by tools: supply metrics to confirm).")

    return {"recommended": bool(reasons), "reasons": reasons, "notes": notes}


# ------------------------------------------------------------------ final report

def build_report(message: str, evidence: list[ToolResult], integration: bool = False) -> str:
    facts: list[str] = []
    uncertainties: list[str] = []
    actions: list[str] = []

    for r in evidence:
        if r.status == "ok":
            facts += [f"[{r.tool}] {ln}" for ln in facts_for(r)]
        elif r.status == "pending":
            actions.append(f"CONFIRMATION REQUIRED to run {r.tool}({r.args}): {r.error}")
        else:
            uncertainties.append(f"{r.tool} {r.status}: {r.error}. Not verified; no value was assumed.")

    got = _by_tool(evidence)
    if "mlflow_find_best_run" in got:
        uncertainties += got["mlflow_find_best_run"].data["caveats"]
    if "k8s_get_deployed_version" in got:
        v = got["k8s_get_deployed_version"].data
        if not v["rollout_complete"]:
            actions.append("Deployment rollout is not complete: inspect Pod events and logs before anything else.")
    if "k8s_find_unhealthy_pods" in got and got["k8s_find_unhealthy_pods"].data["unhealthy_count"]:
        actions.append("Fix the unhealthy Pods (see events/logs above). Retraining will not repair an infrastructure fault.")

    if integration:
        ra = retraining_assessment(evidence, message)
        if ra["recommended"]:
            actions.append("Retraining is recommended because: " + " ".join(ra["reasons"])
                           + " Not started automatically: ask explicitly to trigger the pipeline and confirm.")
        else:
            actions.append("Retraining is NOT recommended on current evidence (no failed quality gate, "
                           "data change or reported degradation found).")
        uncertainties += ra["notes"]

    if not evidence:
        return ("I can help with DVC datasets, MLflow experiments, Airflow pipelines and the Kubernetes "
                "deployment. I did not run any tool for that request, so I have no operational facts to report.")

    out = ["## Known facts (from tool output)"]
    out += [f"- {f}" for f in facts] or ["- None: every tool call failed or is awaiting confirmation."]
    out += ["", "## Uncertainties"] + ([f"- {u}" for u in uncertainties] or ["- None identified."])
    out += ["", "## Recommended actions"] + ([f"- {a}" for a in actions] or ["- None required."])
    out += ["", "## Evidence"]
    out += [f"- {r.tool} -> {r.status} ({r.duration_ms:.0f} ms, request {r.request_id})" for r in evidence]
    return "\n".join(out)
