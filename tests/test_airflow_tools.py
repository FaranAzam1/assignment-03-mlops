from dataclasses import replace

from agent.config import set_settings
from agent.tools import airflow_tools
from agent.tools.base import execute_tool
from tests.conftest import FakeAirflow, Resp, airflow_routes


def test_list_dags_and_runs(fake_airflow):
    dags = execute_tool("airflow_list_dags", {}).data
    assert dags["dags"][0]["dag_id"] == "training_pipeline" and dags["dags"][0]["approved_for_agent"]
    runs = execute_tool("airflow_get_dag_runs", {"dag_id": "training_pipeline"}).data
    assert runs["runs"][0]["state"] == "success"
    # authenticated: token requested first, then used as bearer
    assert fake_airflow.calls[0][1].endswith("/auth/token")


def test_failed_task_and_diagnostics():
    f = FakeAirflow(airflow_routes(state="failed", failed_task="quality_gate"))
    airflow_tools.set_session_factory(f)
    failed = execute_tool("airflow_get_failed_tasks", {}).data
    assert failed["dag_run_id"] == "manual__1"
    assert [t["task_id"] for t in failed["failed_tasks"]] == ["quality_gate"]
    diag = execute_tool("airflow_get_task_diagnostics", {"task_id": "quality_gate"}).data
    assert diag["state"] == "failed" and "below threshold" in diag["error_lines"][-1]


def test_trigger_requires_confirmation_and_approved_dag(fake_airflow):
    res = execute_tool("airflow_trigger_dag", {"dag_id": "training_pipeline"})
    assert res.status == "denied" and "confirmation" in res.error
    assert not any(c[0] == "POST" and "dagRuns" in c[1] for c in fake_airflow.calls)

    bad = execute_tool("airflow_trigger_dag", {"dag_id": "drop_everything"}, confirmed=True)
    assert bad.status == "denied" and "approved" in bad.error

    ok = execute_tool(
        "airflow_trigger_dag",
        {"dag_id": "training_pipeline", "conf": {"f1_threshold": 0.9}},
        confirmed=True,
    )
    assert ok.ok and ok.data["dag_run_id"] == "manual__new"
    assert fake_airflow.calls[-1][2] == {"logical_date": None, "conf": {"f1_threshold": 0.9}}


def test_missing_credentials_is_unavailable(settings):
    set_settings(replace(settings, airflow_password=""))
    assert execute_tool("airflow_list_dags", {}).status == "unavailable"


def test_bad_credentials_reported():
    class Bad(FakeAirflow):
        def post(self, url, json=None, timeout=None):
            return Resp(401, {})

    airflow_tools.set_session_factory(Bad({}))
    res = execute_tool("airflow_list_dags", {})
    assert res.status == "error" and "credentials" in res.error
