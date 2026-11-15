"""Kubernetes tools. Reads are limited to approved namespaces; writes also need confirmation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agent.config import get_settings
from agent.tools.base import PermissionDenied, ToolError, ToolUnavailable, params, tool

_api_factory = None  # tests inject: () -> (core_v1, apps_v1)


def set_api_factory(factory) -> None:
    global _api_factory
    _api_factory = factory


def _apis():
    if _api_factory is not None:
        return _api_factory()
    try:
        from kubernetes import client, config
    except ImportError as exc:  # pragma: no cover
        raise ToolUnavailable("kubernetes python client not installed") from exc
    try:
        config.load_kube_config()
    except Exception as exc:  # noqa: BLE001
        raise ToolUnavailable(f"no usable kubeconfig / cluster: {exc}") from exc
    cfg = client.Configuration.get_default_copy()
    cfg.retries = 0
    client.Configuration.set_default(cfg)
    return client.CoreV1Api(), client.AppsV1Api()


def _ns(namespace: str | None) -> str:
    s = get_settings()
    ns = namespace or s.k8s_namespace
    if ns not in s.allowed_namespaces:
        raise PermissionDenied(f"namespace '{ns}' is not approved; allowed: {list(s.allowed_namespaces)}")
    return ns


def _age(ts) -> str | None:
    if ts is None:
        return None
    secs = int((datetime.now(timezone.utc) - ts).total_seconds())
    if secs < 120:
        return f"{secs}s"
    if secs < 7200:
        return f"{secs // 60}m"
    return f"{secs // 3600}h"


def _pod_view(pod) -> dict[str, Any]:
    statuses = pod.status.container_statuses or []
    containers = []
    for cs in statuses:
        state = cs.state
        reason = None
        if state is not None:
            if getattr(state, "waiting", None):
                reason = state.waiting.reason
            elif getattr(state, "terminated", None):
                reason = state.terminated.reason
        last = getattr(cs, "last_state", None)
        last_reason = None
        if last is not None and getattr(last, "terminated", None):
            last_reason = last.terminated.reason
        containers.append(
            {
                "name": cs.name,
                "ready": cs.ready,
                "restarts": cs.restart_count,
                "image": cs.image,
                "reason": reason,
                "last_terminated_reason": last_reason,
            }
        )
    return {
        "name": pod.metadata.name,
        "phase": pod.status.phase,
        "ready": bool(statuses) and all(c["ready"] for c in containers),
        "restarts": sum(c["restarts"] for c in containers),
        "containers": containers,
        "node": getattr(pod.spec, "node_name", None),
        "age": _age(getattr(pod.metadata, "creation_timestamp", None)),
    }


def _unhealthy_reason(p: dict[str, Any]) -> str | None:
    for c in p["containers"]:  # container-level reasons are more specific than the phase
        if c["reason"] and c["reason"] not in ("Completed",):
            return f"{c['name']}: {c['reason']}"
        if c["restarts"] > 0:
            return f"{c['name']}: restarted {c['restarts']}x (last: {c['last_terminated_reason']})"
        if p["phase"] == "Running" and not c["ready"]:
            return f"{c['name']}: not ready (readiness probe failing?)"
    if p["phase"] not in ("Running", "Succeeded"):
        return f"phase={p['phase']}"
    return None


_NS = {"type": "string", "description": "Namespace; must be approved. Defaults to the app's."}


@tool("k8s_list_pods", "kubernetes", "List application Pods with phase, readiness and restarts.", params({"namespace": _NS}))
def k8s_list_pods(namespace: str | None = None) -> dict[str, Any]:
    ns = _ns(namespace)
    pods = [_pod_view(p) for p in _apis()[0].list_namespaced_pod(ns).items]
    return {"namespace": ns, "count": len(pods), "pods": pods}


@tool(
    "k8s_list_deployments",
    "kubernetes",
    "List Deployments with desired/ready/available replicas and image.",
    params({"namespace": _NS}),
)
def k8s_list_deployments(namespace: str | None = None) -> dict[str, Any]:
    ns = _ns(namespace)
    out = []
    for d in _apis()[1].list_namespaced_deployment(ns).items:
        st = d.status
        out.append(
            {
                "name": d.metadata.name,
                "desired": d.spec.replicas,
                "ready": st.ready_replicas or 0,
                "available": st.available_replicas or 0,
                "updated": st.updated_replicas or 0,
                "images": [c.image for c in d.spec.template.spec.containers],
            }
        )
    return {"namespace": ns, "count": len(out), "deployments": out}


@tool("k8s_list_services", "kubernetes", "Inspect Services: type, cluster IP and ports.", params({"namespace": _NS}))
def k8s_list_services(namespace: str | None = None) -> dict[str, Any]:
    ns = _ns(namespace)
    out = []
    for s in _apis()[0].list_namespaced_service(ns).items:
        out.append(
            {
                "name": s.metadata.name,
                "type": s.spec.type,
                "cluster_ip": s.spec.cluster_ip,
                "selector": s.spec.selector,
                "ports": [
                    {"port": p.port, "target_port": str(p.target_port), "protocol": p.protocol}
                    for p in (s.spec.ports or [])
                ],
            }
        )
    return {"namespace": ns, "count": len(out), "services": out}


@tool(
    "k8s_get_pod_logs",
    "kubernetes",
    "Fetch the last log lines of a Pod (set previous=true for the crashed container).",
    params(
        {
            "pod": {"type": "string"},
            "namespace": _NS,
            "tail_lines": {"type": "integer", "minimum": 1, "maximum": 500},
            "previous": {"type": "boolean"},
        },
        ["pod"],
    ),
)
def k8s_get_pod_logs(
    pod: str, namespace: str | None = None, tail_lines: int = 40, previous: bool = False
) -> dict[str, Any]:
    ns = _ns(namespace)
    try:
        text = _apis()[0].read_namespaced_pod_log(pod, ns, tail_lines=tail_lines, previous=previous)
    except Exception as exc:  # noqa: BLE001
        status = getattr(exc, "status", None)
        if status in (400, 404):  # no logs yet (e.g. ImagePullBackOff) or no such pod
            return {"pod": pod, "namespace": ns, "lines": [], "note": f"no logs available (HTTP {status})"}
        raise
    lines = (text or "").splitlines()
    return {"pod": pod, "namespace": ns, "previous": previous, "lines": lines[-tail_lines:]}


@tool(
    "k8s_get_events",
    "kubernetes",
    "List recent Kubernetes events (Warnings first): probe failures, pull errors, OOM kills.",
    params({"namespace": _NS, "pod": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}),
)
def k8s_get_events(namespace: str | None = None, pod: str | None = None, limit: int = 20) -> dict[str, Any]:
    ns = _ns(namespace)
    kwargs = {"field_selector": f"involvedObject.name={pod}"} if pod else {}
    events = _apis()[0].list_namespaced_event(ns, **kwargs).items

    def when(e):
        return e.last_timestamp or getattr(e, "event_time", None) or e.metadata.creation_timestamp

    events = sorted(events, key=lambda e: when(e) or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    view = [
        {
            "type": e.type,
            "reason": e.reason,
            "object": e.involved_object.name,
            "message": (e.message or "")[:240],
            "count": e.count,
            "age": _age(when(e)),
        }
        for e in events[:limit]
    ]
    view.sort(key=lambda v: v["type"] != "Warning")
    return {"namespace": ns, "count": len(view), "events": view}


@tool(
    "k8s_find_unhealthy_pods",
    "kubernetes",
    "Identify unhealthy Pods (not ready, restarting, crash/pull errors) and recent Warning events.",
    params({"namespace": _NS}),
)
def k8s_find_unhealthy_pods(namespace: str | None = None) -> dict[str, Any]:
    ns = _ns(namespace)
    core = _apis()[0]
    unhealthy = []
    for pod in core.list_namespaced_pod(ns).items:
        view = _pod_view(pod)
        reason = _unhealthy_reason(view)
        if reason:
            unhealthy.append({**view, "problem": reason})
    warnings = [e for e in k8s_get_events(ns, limit=30)["events"] if e["type"] == "Warning"]
    return {
        "namespace": ns,
        "unhealthy_count": len(unhealthy),
        "unhealthy_pods": unhealthy,
        "warning_events": warnings[:10],
    }


@tool(
    "k8s_get_deployed_version",
    "kubernetes",
    "Report the currently deployed application version: image tag, version label, MODEL_VERSION "
    "from the ConfigMap and rollout status.",
    params({"namespace": _NS, "deployment": {"type": "string"}}),
)
def k8s_get_deployed_version(namespace: str | None = None, deployment: str | None = None) -> dict[str, Any]:
    ns = _ns(namespace)
    name = deployment or get_settings().k8s_deployment
    core, apps = _apis()
    try:
        dep = apps.read_namespaced_deployment(name, ns)
    except Exception as exc:  # noqa: BLE001
        if getattr(exc, "status", None) == 404:
            raise ToolError(f"deployment '{name}' not found in namespace '{ns}'") from exc
        raise
    images = [c.image for c in dep.spec.template.spec.containers]
    labels = dep.metadata.labels or {}
    st = dep.status
    cm_version = None
    try:
        cm = core.read_namespaced_config_map("ml-inference-config", ns)
        cm_version = (cm.data or {}).get("MODEL_VERSION")
    except Exception:  # noqa: BLE001 - ConfigMap is optional evidence
        pass
    desired = dep.spec.replicas or 0
    ready = st.ready_replicas or 0
    return {
        "deployment": name,
        "namespace": ns,
        "images": images,
        "image_tags": [i.rsplit(":", 1)[-1] if ":" in i.rsplit("/", 1)[-1] else "latest" for i in images],
        "version_label": labels.get("app.kubernetes.io/version"),
        "model_version_configmap": cm_version,
        "desired_replicas": desired,
        "ready_replicas": ready,
        "updated_replicas": st.updated_replicas or 0,
        "rollout_complete": desired > 0 and ready == desired and (st.updated_replicas or 0) == desired,
    }


@tool(
    "k8s_restart_deployment",
    "kubernetes",
    "Rolling-restart a Deployment. Changes cluster state: approved namespace and explicit "
    "user confirmation required.",
    params({"deployment": {"type": "string"}, "namespace": _NS}, ["deployment"]),
    mutating=True,
)
def k8s_restart_deployment(deployment: str, namespace: str | None = None, _confirmed: bool = False) -> dict[str, Any]:
    ns = _ns(namespace)
    if not _confirmed:
        raise PermissionDenied("restarting a Deployment requires explicit user confirmation")
    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {"kubectl.kubernetes.io/restartedAt": datetime.now(timezone.utc).isoformat()}
                }
            }
        }
    }
    _apis()[1].patch_namespaced_deployment(deployment, ns, body)
    return {"restarted": deployment, "namespace": ns}


@tool(
    "k8s_scale_deployment",
    "kubernetes",
    "Scale a Deployment to N replicas (0-10). Changes cluster state: confirmation required.",
    params(
        {
            "deployment": {"type": "string"},
            "replicas": {"type": "integer", "minimum": 0, "maximum": 10},
            "namespace": _NS,
        },
        ["deployment", "replicas"],
    ),
    mutating=True,
)
def k8s_scale_deployment(
    deployment: str, replicas: int, namespace: str | None = None, _confirmed: bool = False
) -> dict[str, Any]:
    ns = _ns(namespace)
    if not _confirmed:
        raise PermissionDenied("scaling a Deployment requires explicit user confirmation")
    _apis()[1].patch_namespaced_deployment_scale(deployment, ns, {"spec": {"replicas": replicas}})
    return {"scaled": deployment, "namespace": ns, "replicas": replicas}
