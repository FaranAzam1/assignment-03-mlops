from agent.tools import mlflow_tools
from agent.tools.base import execute_tool
from tests.conftest import RUN_A, RUN_B, FakeMlflow, make_run


def test_best_run_picks_highest_f1_and_is_never_marked_production_ready(fake_mlflow):
    res = execute_tool("mlflow_find_best_run", {"metric": "f1", "limit": 5})
    assert res.ok
    assert res.data["best"]["run_id"] == RUN_A
    assert [r["run_id"] for r in res.data["ranking"]] == [RUN_A, RUN_B]
    assert res.data["production_ready"] is False
    assert any("alone" in c for c in res.data["caveats"])


def test_best_run_warns_when_gate_not_passed():
    client = FakeMlflow([make_run(RUN_A, 0.95, gate="failed"), make_run(RUN_B, 0.90)])
    mlflow_tools.set_client_factory(lambda: client)
    res = execute_tool("mlflow_find_best_run", {})
    assert any("NOT cleared" in c for c in res.data["caveats"])


def test_failed_runs_are_ignored():
    client = FakeMlflow([make_run(RUN_A, 0.99, status="FAILED"), make_run(RUN_B, 0.80)])
    mlflow_tools.set_client_factory(lambda: client)
    assert execute_tool("mlflow_find_best_run", {}).data["best"]["run_id"] == RUN_B


def test_no_runs_with_metric_is_error(fake_mlflow):
    res = execute_tool("mlflow_find_best_run", {"metric": "nonexistent"})
    assert res.status == "error" and "no finished runs" in res.error


def test_unknown_experiment(fake_mlflow):
    res = execute_tool("mlflow_list_runs", {"experiment": "nope"})
    assert res.status == "error" and "not found" in res.error


def test_run_details_registry_and_compare(fake_mlflow):
    d = execute_tool("mlflow_get_run_details", {"run_id": RUN_A}).data
    assert d["artifacts"] == ["model.joblib", "scaler.joblib"] and d["dvc_md5"]
    reg = execute_tool("mlflow_get_model_registry", {}).data
    assert reg["models"][0]["versions"][0]["aliases"] == ["candidate"]
    cmp_ = execute_tool("mlflow_compare_runs", {"run_ids": [RUN_A, RUN_B]}).data
    assert len(cmp_["runs"]) == 2 and "f1" in cmp_["metric_names"]
    assert execute_tool("mlflow_compare_runs", {"run_ids": [RUN_A]}).status == "invalid"


def test_mlflow_down_reports_unavailable():
    client = FakeMlflow(fail=ConnectionError("Max retries exceeded: connection refused"))
    mlflow_tools.set_client_factory(lambda: client)
    res = execute_tool("mlflow_list_experiments", {})
    assert res.status == "unavailable" and res.data is None
