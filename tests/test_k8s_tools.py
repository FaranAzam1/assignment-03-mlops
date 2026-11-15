from agent.tools.base import execute_tool
from tests.conftest import make_pod


def test_healthy_deployment_version(healthy_k8s):
    v = execute_tool("k8s_get_deployed_version", {}).data
    assert v["image_tags"] == ["1.0.0"] and v["rollout_complete"]
    assert v["model_version_configmap"] == "1.0.0"
    assert execute_tool("k8s_find_unhealthy_pods", {}).data["unhealthy_count"] == 0


def test_unhealthy_pods_and_events_detected(broken_k8s):
    d = execute_tool("k8s_find_unhealthy_pods", {}).data
    assert d["unhealthy_count"] == 2
    assert "ImagePullBackOff" in d["unhealthy_pods"][0]["problem"]
    assert d["warning_events"][0]["reason"] == "Failed"
    assert not execute_tool("k8s_get_deployed_version", {}).data["rollout_complete"]


def test_restarting_pod_is_flagged(healthy_k8s):
    healthy_k8s[0].pods = [make_pod("ml-inference-1", restarts=4, last_term="OOMKilled")]
    d = execute_tool("k8s_find_unhealthy_pods", {}).data
    assert "OOMKilled" in d["unhealthy_pods"][0]["problem"]


def test_logs_services_deployments(healthy_k8s):
    assert execute_tool("k8s_get_pod_logs", {"pod": "ml-inference-1"}).data["lines"][-1] == "ERROR boom"
    assert execute_tool("k8s_list_services", {}).data["services"][0]["type"] == "ClusterIP"
    assert execute_tool("k8s_list_deployments", {}).data["deployments"][0]["desired"] == 2


def test_unapproved_namespace_rejected_for_reads_and_writes(healthy_k8s):
    assert execute_tool("k8s_list_pods", {"namespace": "kube-system"}).status == "denied"
    res = execute_tool(
        "k8s_restart_deployment", {"deployment": "x", "namespace": "kube-system"}, confirmed=True
    )
    assert res.status == "denied" and "not approved" in res.error
    assert healthy_k8s[1].patches == []


def test_modification_without_confirmation_is_rejected(healthy_k8s):
    res = execute_tool("k8s_restart_deployment", {"deployment": "ml-inference"})
    assert res.status == "denied" and "confirmation" in res.error
    res = execute_tool("k8s_scale_deployment", {"deployment": "ml-inference", "replicas": 5})
    assert res.status == "denied"
    assert healthy_k8s[1].patches == []


def test_confirmed_modifications_patch_cluster(healthy_k8s):
    assert execute_tool("k8s_restart_deployment", {"deployment": "ml-inference"}, confirmed=True).ok
    scale = {"deployment": "ml-inference", "replicas": 3}
    assert execute_tool("k8s_scale_deployment", scale, confirmed=True).ok
    assert len(healthy_k8s[1].patches) == 2
    assert healthy_k8s[1].patches[1][2] == {"spec": {"replicas": 3}}


def test_scale_bounds_validated(healthy_k8s):
    res = execute_tool("k8s_scale_deployment", {"deployment": "d", "replicas": 500}, confirmed=True)
    assert res.status == "invalid"
