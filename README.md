# Assignment 03: Production-Ready MLOps AI Agent

A conversational agent that answers MLOps questions by calling real tools over **DVC**, **MLflow**,
**Apache Airflow 3** and **Kubernetes**, and reports only what the tools returned. Local LLM: **Ollama**.
Design rationale: [architecture.md](architecture.md). Report: [report/report.md](report/report.md) / `report.pdf`.

## Prerequisites (Windows)

| Tool | Notes |
|---|---|
| Python 3.10+ | 3.12/3.14 tested |
| Docker Desktop | with **Kubernetes enabled** (Settings > Kubernetes > Enable, kubeadm cluster) |
| Ollama | `ollama pull llama3.2:1b` (any tool-capable model works) |
| Git, kubectl | kubectl ships with Docker Desktop |

## 1. Install

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,report]"     # Linux/macOS: .venv/bin/python
python scripts/setup_env.py     # random local credentials -> .env + airflow-passwords.json (git-ignored)
```

Environment variables (see `.env.example`): `OLLAMA_URL`, `OLLAMA_MODEL`, `MLFLOW_TRACKING_URI`,
`AIRFLOW_URL`, `AIRFLOW_USERNAME`, `AIRFLOW_PASSWORD`, `K8S_NAMESPACE`, `ALLOWED_NAMESPACES`,
`ALLOWED_DAG_IDS`, `TOOL_TIMEOUT_S`. **No secret is committed**; `.env` is ignored.

## 2. Data (DVC)

```bash
dvc pull                         # restore data/dataset.csv from the DVC remote (local .dvc-storage)
python scripts/make_dataset.py v2 && dvc add data/dataset.csv   # how a new version is made
```

Two versions exist in Git history, tagged `data-v1` (1000 rows) and `data-v2` (2000 rows).
Reproduce an old one: `git checkout data-v1 -- data/dataset.csv.dvc && dvc checkout data/dataset.csv`.
The DVC remote is a local folder (`.dvc-storage/`, not in Git); point it at S3/GCS for real use.

## 3. Start the services

```bash
docker compose up -d --build     # MLflow :5000, Airflow 3 UI/API :8080
```

Airflow login is `admin` / the `AIRFLOW_PASSWORD` in your `.env`. The DAG `training_pipeline`
appears after ~30 s (DAGs are unpaused by default for API triggers; unpause in the UI if needed).

**Kubernetes** (after training has produced `models/model.joblib`):

```bash
docker build -t ml-inference:1.0.0 .
kubectl apply -f kubernetes/
kubectl -n mlops get pods,svc
kubectl -n mlops port-forward svc/ml-inference 8000:80   # then GET http://localhost:8000/ready
```

**Train once without Airflow** (also writes `models/model.joblib`):

```bash
python -m training.train --threshold 0.8
```

## 4. Talk to the agent

CLI:

```bash
python -m agent.main "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
python -m agent.main            # interactive
python -m agent.main --no-llm "why is my prediction api unavailable?"   # deterministic planner only
```

HTTP API:

```bash
uvicorn api.main:app --port 8001
curl -s localhost:8001/tools
curl -s -X POST localhost:8001/chat -H "content-type: application/json" \
     -d '{"message": "which dataset version was used for the best run?"}'
curl -s -X POST localhost:8001/confirm -H "content-type: application/json" -d '{"action_id": "<id from /chat>"}'
```

State-changing requests ("trigger the training pipeline", "restart the deployment", "scale to 3
replicas") return `pending_actions`; nothing happens until `/confirm` (CLI asks `Confirm? [y/N]`).

Example questions: `Which dataset version did the best run use?` · `Compare the last five training runs and
recommend the best candidate by F1` · `Why did the last pipeline run fail?` · `Why is my prediction API
unavailable?` · `Trigger the model training pipeline` (asks for confirmation).

## 5. Tests and lint

```bash
pytest                 # 64 tests; every external system is mocked, none needs a running service
ruff check .
```

CI (`.github/workflows/ci.yml`) runs ruff and pytest on every pull request and push to `main`.

## 6. Failure injection

| Layer | Command | Expected diagnosis |
|---|---|---|
| Airflow, data | trigger `training_pipeline` with conf `{"inject_failure": "validation"}` | `validate_dataset` failed, downstream `upstream_failed`, no model |
| Airflow, model | conf `{"f1_threshold": 0.9999}` | `quality_gate` failed, run tagged `failed`, nothing registered/exported |
| Kubernetes, image | `python scripts/inject_k8s_failure.py bad-image` | `ImagePullBackOff`, 0 ready |
| Kubernetes, probe | `python scripts/inject_k8s_failure.py bad-probe` | Pods Running but not Ready, readiness 404 |
| Restore | `python scripts/inject_k8s_failure.py restore` | rollout completes |

## Layout

```
agent/        main.py orchestrator.py planner.py reporting.py llm.py schemas.py tools/{dvc,mlflow,airflow,kubernetes}_tools.py
api/          FastAPI interface for the agent
inference/    FastAPI model service that runs in Kubernetes (root Dockerfile)
dags/         training_pipeline.py (Airflow 3 DAG)
training/     train.py  (stage functions shared by DAG, CLI, tests)
kubernetes/   namespace, configmap, deployment, service
tests/  scripts/  screenshots/  report/
docker-compose.yml  Dockerfile  Dockerfile.airflow  Dockerfile.mlflow
```

Note: the brief's structure lists a top-level `dvc.yaml`. Data versioning here uses
`data/dataset.csv.dvc` (one tracked file, no pipeline stages), so no `dvc.yaml` is needed.

## Known limitations

See the report (section 8). Main ones: a 1B local model needs the deterministic planner for multi-tool
requests; the DVC remote is local; Airflow runs in standalone mode with SQLite (demo, not HA).
