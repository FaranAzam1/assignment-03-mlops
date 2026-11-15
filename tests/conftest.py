"""Shared fakes. No test touches a real MLflow, Airflow, Kubernetes, Ollama or network."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from agent.config import Settings, set_settings
from agent.tools import airflow_tools, kubernetes_tools, mlflow_tools

NOW = datetime.now(timezone.utc)
RUN_A = "a" * 32
RUN_B = "b" * 32
MD5_V2 = "2" * 32


@pytest.fixture(autouse=True)
def settings(tmp_path):
    s = Settings(repo_root=tmp_path, log_file=tmp_path / "logs" / "agent.jsonl",
                 airflow_password="test-pw", tool_timeout_s=2.0)
    set_settings(s)
    airflow_tools.set_session_factory(__import__("requests").Session)
    mlflow_tools.set_client_factory(None)
    kubernetes_tools.set_api_factory(None)
    yield s
    set_settings(None)
    mlflow_tools.set_client_factory(None)
    kubernetes_tools.set_api_factory(None)


def read_log(settings) -> list[dict]:
    if not settings.log_file.exists():
        return []
    return [json.loads(line) for line in settings.log_file.read_text().splitlines()]


# ----------------------------------------------------------------------------- MLflow
def make_run(run_id, f1, status="FINISHED", gate="passed", md5=MD5_V2, start=1_700_000_000_000):
    return NS(
        info=NS(run_id=run_id, status=status, start_time=start),
        data=NS(metrics={"f1": f1, "accuracy": f1 - 0.01},
                params={"model_type": "random_forest"},
                tags={"quality_gate": gate, "dvc.md5": md5, "dvc.path": "data/dataset.csv",
                      "dvc.rows": "2000", "mlflow.runName": f"run-{run_id[:2]}"}),
    )


class FakeMlflow:
    def __init__(self, runs=None, fail=None):
        self.runs = runs if runs is not None else [make_run(RUN_A, 0.91), make_run(RUN_B, 0.88)]
        self.fail = fail

    def _check(self):
        if self.fail:
            raise self.fail

    def get_experiment_by_name(self, name):
        self._check()
        return NS(experiment_id="1", name=name) if name == "training_pipeline" else None

    def get_experiment(self, exp_id):
        raise KeyError(exp_id)

    def search_runs(self, ids, order_by=None, max_results=10):
        self._check()
        return self.runs[:max_results]

    def get_run(self, run_id):
        self._check()
        for r in self.runs:
            if r.info.run_id == run_id:
                return r
        raise KeyError(run_id)

    def list_artifacts(self, run_id):
        return [NS(path="model.joblib"), NS(path="scaler.joblib")]

    def search_experiments(self):
        self._check()
        return [NS(experiment_id="1", name="training_pipeline", lifecycle_stage="active")]

    def search_registered_models(self):
        return [NS(name="ml-inference-model", aliases={"candidate": "1"})]

    def search_model_versions(self, f):
        return [NS(version="1", run_id=RUN_A, status="READY", aliases=["candidate"])]


@pytest.fixture
def fake_mlflow():
    client = FakeMlflow()
    mlflow_tools.set_client_factory(lambda: client)
    return client


# ----------------------------------------------------------------------------- Airflow
class Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body, self.text = status, body or {}, json.dumps(body or {})

    def json(self):
        return self._body


class FakeAirflow:
    """Callable factory + recorder. routes: {(METHOD, path_suffix): Resp or callable}."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self):
        return self

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        return Resp(200, {"access_token": "jwt-token-value"})

    def request(self, method, url, headers=None, timeout=None, params=None, json=None):
        self.calls.append((method, url, json))
        path = url.split("/api/v2", 1)[1]
        for (m, suffix), resp in self.routes.items():
            if m == method and path.endswith(suffix):
                return resp(path) if callable(resp) else resp
        return Resp(404, {"detail": "not found"})


def airflow_routes(state="success", failed_task=None):
    tasks = [{"task_id": t, "state": "success", "try_number": 1, "max_tries": 2}
             for t in ("validate_dataset", "preprocess_data", "train_model", "evaluate_and_log")]
    tasks.append({"task_id": "quality_gate", "state": "success", "try_number": 1, "max_tries": 0})
    if failed_task:
        for t in tasks:
            if t["task_id"] == failed_task:
                t["state"] = "failed"
    return {
        ("GET", "/dags/training_pipeline/dagRuns"): Resp(200, {"dag_runs": [
            {"dag_run_id": "manual__1", "state": state, "run_type": "manual",
             "start_date": "2026-11-15T10:00:00Z", "end_date": None, "conf": {}}]}),
        ("GET", "/dags/training_pipeline/dagRuns/manual__1/taskInstances"): Resp(200, {"task_instances": tasks}),
        ("GET", "/dags/training_pipeline/dagRuns/manual__1/taskInstances/quality_gate"): Resp(200, {
            "state": "failed", "try_number": 1, "max_tries": 0, "duration": 1.2}),
        ("GET", "/taskInstances/quality_gate/logs/1"): Resp(200, {"content": [
            "INFO starting", "ERROR quality gate failed: f1=0.7000 below threshold 0.9900"]}),
        ("POST", "/dags/training_pipeline/dagRuns"): Resp(200, {
            "dag_run_id": "manual__new", "state": "queued", "run_type": "manual"}),
        ("GET", "/dags"): Resp(200, {"dags": [{"dag_id": "training_pipeline", "is_paused": False,
                                                 "tags": [{"name": "mlops"}]}]}),
    }


@pytest.fixture
def fake_airflow():
    f = FakeAirflow(airflow_routes())
    airflow_tools.set_session_factory(f)
    return f


# ----------------------------------------------------------------------------- Kubernetes
def make_pod(name, ready=True, phase="Running", restarts=0, waiting=None, last_term=None):
    state = NS(waiting=NS(reason=waiting) if waiting else None, terminated=None)
    last = NS(terminated=NS(reason=last_term) if last_term else None)
    cs = NS(name="api", ready=ready, restart_count=restarts, image="ml-inference:1.0.0",
            state=state, last_state=last)
    return NS(metadata=NS(name=name, creation_timestamp=NOW),
              status=NS(phase=phase, container_statuses=[cs]), spec=NS(node_name="node1"))


def make_deployment(name="ml-inference", replicas=2, ready=2, image="ml-inference:1.0.0"):
    return NS(metadata=NS(name=name, labels={"app.kubernetes.io/version": "1.0.0"}),
              spec=NS(replicas=replicas, template=NS(spec=NS(containers=[NS(image=image)]))),
              status=NS(ready_replicas=ready, available_replicas=ready, updated_replicas=ready))


class FakeCore:
    def __init__(self, pods, events=None, logs="INFO started\nERROR boom"):
        self.pods, self.events, self.logs = pods, events or [], logs

    def list_namespaced_pod(self, ns):
        return NS(items=self.pods)

    def list_namespaced_event(self, ns, **kw):
        return NS(items=self.events)

    def list_namespaced_service(self, ns):
        return NS(items=[NS(metadata=NS(name="ml-inference"),
                            spec=NS(type="ClusterIP", cluster_ip="10.0.0.1", selector={"app": "ml-inference"},
                                    ports=[NS(port=80, target_port=8000, protocol="TCP")]))])

    def read_namespaced_pod_log(self, pod, ns, tail_lines=40, previous=False):
        return self.logs

    def read_namespaced_config_map(self, name, ns):
        return NS(data={"MODEL_VERSION": "1.0.0"})


class FakeApps:
    def __init__(self, dep):
        self.dep, self.patches = dep, []

    def list_namespaced_deployment(self, ns):
        return NS(items=[self.dep])

    def read_namespaced_deployment(self, name, ns):
        return self.dep

    def patch_namespaced_deployment(self, name, ns, body):
        self.patches.append((name, ns, body))

    def patch_namespaced_deployment_scale(self, name, ns, body):
        self.patches.append((name, ns, body))


def make_event(reason, obj, msg, typ="Warning"):
    return NS(type=typ, reason=reason, message=msg, count=3, last_timestamp=NOW, event_time=None,
              metadata=NS(creation_timestamp=NOW), involved_object=NS(name=obj))


@pytest.fixture
def healthy_k8s():
    apps = FakeApps(make_deployment())
    core = FakeCore([make_pod("ml-inference-1"), make_pod("ml-inference-2")])
    kubernetes_tools.set_api_factory(lambda: (core, apps))
    return core, apps


@pytest.fixture
def broken_k8s():
    apps = FakeApps(make_deployment(ready=0, image="ml-inference:9.9.9"))
    pods = [make_pod("ml-inference-x", ready=False, phase="Pending", waiting="ImagePullBackOff"),
            make_pod("ml-inference-y", ready=False, phase="Pending", waiting="ImagePullBackOff")]
    events = [make_event("Failed", "ml-inference-x", "Failed to pull image ml-inference:9.9.9: not found")]
    core = FakeCore(pods, events)
    kubernetes_tools.set_api_factory(lambda: (core, apps))
    return core, apps


def repo_path(tmp_path: Path) -> Path:
    return tmp_path
