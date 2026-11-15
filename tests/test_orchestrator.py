"""Agent behaviour: tool selection, grounding, permissions, multi-tool integration."""

import subprocess

from agent.llm import LLMError, LLMResponse
from agent.orchestrator import Agent
from agent.tools import airflow_tools, kubernetes_tools, mlflow_tools
from tests.conftest import (
    MD5_V2,
    RUN_A,
    RUN_B,
    FakeAirflow,
    FakeApps,
    FakeCore,
    FakeMlflow,
    airflow_routes,
    make_deployment,
    make_pod,
    make_run,
)


class StubLLM:
    """Scripted LLM: returns queued responses, records what it was asked."""

    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def call(name, **args):
    return LLMResponse(tool_calls=[{"name": name, "arguments": args}])


# ------------------------------------------------------------------ selection + execution
def test_llm_selects_and_executes_tool(fake_mlflow):
    llm = StubLLM(call("mlflow_find_best_run", metric="f1"), LLMResponse(content="done"),
                  LLMResponse(content="Best run is aaaaaaaa."))
    resp = Agent(llm=llm).handle("which mlflow run has the best f1?")
    assert resp.mode == "llm-tools"
    assert [t.tool for t in resp.tool_calls] == ["mlflow_find_best_run"]
    assert RUN_A[:8] in resp.answer
    # the LLM was only offered MLflow tools for an MLflow question
    offered = {t["function"]["name"] for t in llm.calls[0][1]}
    assert offered and all(n.startswith("mlflow_") for n in offered)


def test_llm_cannot_inject_confirmation_flag(healthy_k8s):
    llm = StubLLM(call("k8s_restart_deployment", deployment="ml-inference", _confirmed=True),
                  LLMResponse(content="x"))
    resp = Agent(llm=llm, llm_summary=False).handle("please restart the kubernetes deployment")
    assert healthy_k8s[1].patches == []
    assert resp.tool_calls[0].status == "pending"


def test_llm_choosing_nothing_falls_back_to_rules(fake_mlflow):
    llm = StubLLM(LLMResponse(content="I think the best model is magic-123."))
    resp = Agent(llm=llm, llm_summary=False).handle("what is the best mlflow model by f1?")
    assert resp.mode == "rules-fallback"
    assert "magic-123" not in resp.answer and RUN_A[:8] in resp.answer


def test_llm_down_falls_back_to_rules(fake_mlflow):
    llm = StubLLM(LLMError("Ollama unreachable"), LLMError("Ollama unreachable"))
    resp = Agent(llm=llm).handle("list mlflow experiments")
    assert resp.mode == "rules-fallback" and resp.tool_calls[0].ok


def test_no_llm_uses_deterministic_planner(fake_mlflow):
    resp = Agent(llm=None).handle("list mlflow experiments")
    assert resp.mode == "rules" and "training_pipeline" in resp.answer


def test_out_of_scope_request_runs_no_tool():
    resp = Agent(llm=None).handle("tell me a joke")
    assert resp.tool_calls == [] and "no operational facts" in resp.answer


# ------------------------------------------------------------------ no invented results
def test_tool_unavailable_means_no_invented_results():
    mlflow_tools.set_client_factory(
        lambda: FakeMlflow(fail=ConnectionError("Connection refused")))
    resp = Agent(llm=None).handle("what is the best mlflow run by f1?")
    assert resp.tool_calls[0].status == "unavailable"
    assert "Not verified" in resp.answer and "None: every tool call failed" in resp.answer
    assert "run " not in resp.answer.split("## Uncertainties")[0].lower().replace("## known facts", "")


def test_all_services_down_integration_reports_everything_unverified():
    mlflow_tools.set_client_factory(lambda: FakeMlflow(fail=ConnectionError("refused")))
    airflow_tools.set_session_factory(lambda: (_ for _ in ()).throw(ConnectionError("refused")))
    kubernetes_tools.set_api_factory(lambda: (_ for _ in ()).throw(ConnectionError("refused")))
    resp = Agent(llm=None).handle(
        "Find the best recent model, identify the dataset version used to train it, check the "
        "deployment status on kubernetes, and explain whether retraining is necessary.")
    assert resp.tool_calls and all(not t.ok for t in resp.tool_calls)
    assert "Retraining is NOT recommended" in resp.answer  # no evidence => no recommendation to retrain
    assert "could not be checked" in resp.answer


# ------------------------------------------------------------------ authorization
def test_unauthorized_k8s_change_is_parked_not_executed(healthy_k8s):
    agent = Agent(llm=None)
    resp = agent.handle("restart the kubernetes deployment")
    assert healthy_k8s[1].patches == []
    assert resp.pending_actions and resp.tool_calls[0].status == "pending"
    assert "CONFIRMATION REQUIRED" in resp.answer


def test_confirmation_executes_exactly_once(healthy_k8s):
    agent = Agent(llm=None)
    action = agent.handle("restart the kubernetes deployment").pending_actions[0]
    done = agent.confirm(action.action_id)
    assert done.tool_calls[0].ok and len(healthy_k8s[1].patches) == 1
    again = agent.confirm(action.action_id)
    assert again.tool_calls[0].status == "invalid" and len(healthy_k8s[1].patches) == 1


def test_rejected_action_is_dropped(healthy_k8s):
    agent = Agent(llm=None)
    action = agent.handle("scale the kubernetes deployment to 3 replicas").pending_actions[0]
    assert agent.reject(action.action_id)
    assert agent.confirm(action.action_id).tool_calls[0].status == "invalid"
    assert healthy_k8s[1].patches == []


def test_wrong_namespace_denied_immediately_not_parked(healthy_k8s):
    agent = Agent(llm=None)
    result = agent._run_tool("k8s_restart_deployment",
                             {"deployment": "x", "namespace": "kube-system"}, "req1")
    assert result.status == "denied" and not agent._pending


def test_retraining_is_never_triggered_automatically(fake_airflow):
    resp = Agent(llm=None).handle("did the airflow training pipeline run ok? should we retrain?")
    assert not any(c[0] == "POST" and "dagRuns" in c[1] for c in fake_airflow.calls)
    assert all(t.tool != "airflow_trigger_dag" for t in resp.tool_calls)


def test_triggering_pipeline_needs_confirmation(fake_airflow):
    agent = Agent(llm=None)
    resp = agent.handle("trigger the airflow training pipeline")
    assert resp.pending_actions and not any(c[0] == "POST" and "dagRuns" in c[1] for c in fake_airflow.calls)
    done = agent.confirm(resp.pending_actions[0].action_id)
    assert done.tool_calls[0].ok and "manual__new" in done.answer


# ------------------------------------------------------------------ mandatory integration
INTEGRATION = ("Find the best recent model, identify the dataset version used to train it, check the "
               "deployment status, and explain whether retraining is necessary. (mlflow, dvc, kubernetes, airflow)")


def _setup_world(tmp_path, gate="passed", dag_state="success", failed_task=None, md5=MD5_V2):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@e.com")
    git("config", "user.name", "t")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "dataset.csv.dvc").write_text(f"outs:\n- md5: {MD5_V2}\n  size: 9\n  path: dataset.csv\n")
    git("add", ".")
    git("commit", "-q", "-m", "data v2")
    git("tag", "data-v2")
    client = FakeMlflow([make_run(RUN_A, 0.91, gate=gate, md5=md5), make_run(RUN_B, 0.88)])
    mlflow_tools.set_client_factory(lambda: client)
    airflow_tools.set_session_factory(FakeAirflow(airflow_routes(dag_state, failed_task)))
    apps = FakeApps(make_deployment())
    kubernetes_tools.set_api_factory(
        lambda: (FakeCore([make_pod("ml-inference-1"), make_pod("ml-inference-2")]), apps))
    return apps


def test_integration_scenario_uses_all_four_technologies_and_correlates(tmp_path):
    _setup_world(tmp_path)
    resp = Agent(llm=None).handle(INTEGRATION)
    tools = [t.tool for t in resp.tool_calls]
    assert tools[0] == "mlflow_find_best_run"
    assert {"dvc_get_run_dataset_provenance", "k8s_get_deployed_version", "k8s_find_unhealthy_pods",
            "airflow_get_dag_runs"} <= set(tools)
    assert "airflow_trigger_dag" not in tools
    assert all(t.ok for t in resp.tool_calls), [(t.tool, t.status, t.error) for t in resp.tool_calls]
    prov = next(t for t in resp.tool_calls if t.tool == "dvc_get_run_dataset_provenance")
    assert prov.args["run_id"] == RUN_A  # dependency passed from the MLflow step
    assert prov.data["dataset_version"] == ["data-v2"]
    for section in ("Known facts", "Uncertainties", "Recommended actions", "Evidence"):
        assert section in resp.answer
    assert "Retraining is NOT recommended" in resp.answer


def test_integration_recommends_retraining_only_on_evidence_and_does_not_start_it(tmp_path):
    _setup_world(tmp_path, gate="failed", dag_state="failed", failed_task="quality_gate")
    resp = Agent(llm=None).handle(INTEGRATION)
    assert "Retraining is recommended" in resp.answer and "quality gate" in resp.answer
    assert "airflow_trigger_dag" not in [t.tool for t in resp.tool_calls]
    assert not resp.pending_actions


def test_integration_flags_data_changed_since_training(tmp_path):
    _setup_world(tmp_path, md5="f" * 32)
    resp = Agent(llm=None).handle(INTEGRATION)
    assert "different dataset version" in resp.answer
