# Architecture

```
User / Engineer ──► CLI (agent/main.py)  or  FastAPI (api/main.py: /chat /confirm /tools)
                              │  validation (pydantic)
                              ▼
                     Orchestrator (agent/orchestrator.py)
          ┌───────────────────┼─────────────────────────────┐
          │ planner.py        │ llm.py (Ollama, tool calling)│ permissions + tracing
          │ deterministic     │ picks tools for single-domain│ confirmation gate, structured
          │ workflows         │ requests, writes a summary   │ JSON logs, request ids
          └───────────────────┴─────────────────────────────┘
                              │ execute_tool(): schema validation, timeout, error mapping
      ┌───────────────┬───────┴────────┬────────────────┬──────────────────┐
      ▼               ▼                ▼                ▼
  dvc_tools       mlflow_tools     airflow_tools    kubernetes_tools
  git + dvc CLI   MlflowClient     Airflow 3 REST   kubernetes client
  (allow-listed)  (tracking+reg.)  (/auth/token,     (approved namespace,
                                    /api/v2)         confirm to mutate)
      └───────────────┴────────┬───────┴────────────────┘
                               ▼
                reporting.py: evidence-only answer
        Known facts · Uncertainties · Recommended actions · Evidence
```

## Components

| Path | Role |
|---|---|
| `agent/orchestrator.py` | Agent loop, permission/confirmation gate, strategy selection, logging |
| `agent/planner.py` | Request classification, deterministic multi-tool workflows with data dependencies |
| `agent/llm.py` | Ollama `/api/chat` client with native tool calling |
| `agent/reporting.py` | Builds the answer *only* from tool results; retraining decision rules |
| `agent/tools/base.py` | Tool registry, JSON-schema argument validation, timeout, error classification, command allow-list |
| `agent/tools/*_tools.py` | The four tool modules (7–10 tools each) |
| `dags/training_pipeline.py` | Airflow 3 DAG: validate → preprocess → train → evaluate/log to MLflow → quality gate |
| `training/train.py` | Pipeline stage functions, shared by the DAG, the CLI and the tests |
| `inference/app.py` + `Dockerfile` | FastAPI inference service deployed on Kubernetes |
| `kubernetes/` | Namespace, ConfigMap, Deployment (2 replicas, probes, requests/limits), ClusterIP Service |
| `docker-compose.yml` | MLflow server + Airflow 3 |

## Design decisions and justification

**Why a hybrid planner + LLM.** The local model (`llama3.2:1b`, via Ollama) selects tools reliably for a
single-domain question ("best run by f1") but cannot be trusted to chain four tools with data
dependencies (the best run's id feeds the DVC provenance lookup, which feeds the retraining decision).
Multi-technology requests therefore run through a deterministic workflow, which guarantees the order,
passes arguments between steps, and makes the behaviour testable. The LLM still selects tools for
single-domain requests, and writes a short summary that is labelled as LLM output. If the LLM is down or
selects nothing, the planner takes over (`mode: rules-fallback`), so the agent degrades instead of
guessing.

**Evidence over generation.** The final answer is assembled by `reporting.py` from `ToolResult.data`.
A failed tool contributes an *uncertainty* line ("not verified; no value was assumed"), never a fact.
Every answer ends with an evidence list (tool → status, duration, request id).

**Permissions.**
1. Read tools are limited to the approved namespace(s) and DAG ids (`ALLOWED_NAMESPACES`, `ALLOWED_DAG_IDS`).
2. Three tools mutate state (`k8s_restart_deployment`, `k8s_scale_deployment`, `airflow_trigger_dag`).
   They refuse to run unless the orchestrator passes `_confirmed=True`. That flag is never read from
   LLM arguments (keys starting with `_` are stripped); it is set only by `Agent.confirm()` after the
   user explicitly confirms the parked action. The tool re-checks namespace/DAG at execution time too
   (defence in depth).
3. `git`/`dvc` run through an allow-list of read-only subcommands, `shell=False`.

**Retraining policy.** Retraining is never started by an analysis request. It is *recommended* only on
evidence: a failed quality gate, a best candidate that never passed the gate, a model trained on a
dataset version that differs from the current committed one, or user-reported degradation (flagged as
unverified). With no freshness policy configured, model age alone is not evidence.

**Reproducibility / provenance.** The pipeline tags every MLflow run with `dvc.md5`, `dvc.path`,
`dvc.rows` and `git.commit`. `dvc_get_run_dataset_provenance` resolves that md5 to the Git commit and
`data-vN` tag that introduced it (`git log -S<md5>`), then gives the exact `git checkout` / `dvc checkout`
commands to restore it.

**Quality gate.** `f1 >= threshold` (default 0.80, a DAG param). Pass → run tagged `passed`, model
registered with alias `candidate`, model bundle exported for the inference image. Fail → run tagged
`failed`, `AirflowFailException` (no retries), nothing registered or exported.

**Retries.** Tasks retry twice (10 s delay). Validation and the gate use `retries=0` / `AirflowFailException`
because retrying cannot repair bad data or a bad model.

## Tool catalogue and permissions

| Module | Tool | Mutates | Notes |
|---|---|---|---|
| DVC | `dvc_list_datasets`, `dvc_get_dataset_version`, `dvc_list_dataset_history`, `dvc_compare_dataset_versions`, `dvc_check_data_changed`, `dvc_explain_reproduce`, `dvc_get_run_dataset_provenance` | no | git/dvc read-only allow-list |
| MLflow | `mlflow_list_experiments`, `mlflow_list_runs`, `mlflow_get_run_details`, `mlflow_find_best_run`, `mlflow_compare_runs`, `mlflow_compare_experiments`, `mlflow_get_model_registry` | no | MlflowClient |
| Airflow | `airflow_list_dags`, `airflow_get_dag_runs`, `airflow_get_failed_tasks`, `airflow_get_task_diagnostics` | no | JWT from `/auth/token` |
| Airflow | `airflow_trigger_dag` | **yes** | approved DAG id + confirmation |
| Kubernetes | `k8s_list_pods`, `k8s_list_deployments`, `k8s_list_services`, `k8s_get_pod_logs`, `k8s_get_events`, `k8s_find_unhealthy_pods`, `k8s_get_deployed_version` | no | approved namespace |
| Kubernetes | `k8s_restart_deployment`, `k8s_scale_deployment` | **yes** | approved namespace + confirmation |

## Logging

Each tool call writes one JSON line to `logs/agent.jsonl`:
`{ts, event:"tool_call", request_id, tool, status, duration_ms, args, error}`. Keys matching
`password|token|secret|api_key|authorization|credential` and `Bearer …`/JWT-looking values are redacted;
long strings are truncated.

## Failure modes handled

Service down → `unavailable`; slow call → `timeout`; bad arguments → `invalid` (never executed);
forbidden → `denied`; LLM down → planner fallback; unknown tool name from the LLM → `invalid`.
