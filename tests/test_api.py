from fastapi.testclient import TestClient

from agent.orchestrator import Agent
from api.main import create_app
from tests.conftest import RUN_A


def test_health_and_tool_catalogue():
    client = TestClient(create_app(Agent(llm=None)))
    assert client.get("/health").json() == {"status": "ok"}
    tools = client.get("/tools").json()
    modules = {t["module"] for t in tools}
    assert modules == {"dvc", "mlflow", "airflow", "kubernetes"}
    for m in modules:  # assignment: at least three callable tools per module
        assert sum(t["module"] == m for t in tools) >= 3
    assert {t["name"] for t in tools if t["mutating"]} == {
        "airflow_trigger_dag", "k8s_restart_deployment", "k8s_scale_deployment"}


def test_chat_returns_structured_evidence(fake_mlflow):
    client = TestClient(create_app(Agent(llm=None)))
    body = client.post("/chat", json={"message": "best mlflow run by f1"}).json()
    assert body["tool_calls"][0]["tool"] == "mlflow_find_best_run"
    assert body["tool_calls"][0]["data"]["best"]["run_id"] == RUN_A
    assert body["request_id"] and body["mode"] == "rules"


def test_chat_validates_input():
    client = TestClient(create_app(Agent(llm=None)))
    assert client.post("/chat", json={}).status_code == 422
    assert client.post("/chat", json={"message": ""}).status_code == 422
    assert client.post("/chat", json={"message": "   "}).status_code == 422
    assert client.post("/chat", json={"message": "x" * 5000}).status_code == 422


def test_confirm_flow_over_http(healthy_k8s):
    client = TestClient(create_app(Agent(llm=None)))
    chat = client.post("/chat", json={"message": "restart the kubernetes deployment"}).json()
    action_id = chat["pending_actions"][0]["action_id"]
    assert healthy_k8s[1].patches == []
    done = client.post("/confirm", json={"action_id": action_id}).json()
    assert done["tool_calls"][0]["status"] == "ok" and len(healthy_k8s[1].patches) == 1
    assert client.post("/confirm", json={"action_id": "zzzzzzzz"}).json()["tool_calls"][0]["status"] == "invalid"
