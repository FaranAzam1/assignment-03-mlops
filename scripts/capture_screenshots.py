"""Capture screenshots of the MLflow and Airflow UIs plus terminal-style renders of CLI evidence.

Needs Playwright and a local Edge/Chrome (not part of the project's requirements):
    pip install playwright && python scripts/capture_screenshots.py
Credentials come from .env (AIRFLOW_USERNAME / AIRFLOW_PASSWORD); nothing is written to the images
except what the UIs show.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "screenshots"
EVID = ROOT / "report" / "evidence"
load_dotenv(ROOT / ".env")

MLFLOW = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
AIRFLOW = os.environ.get("AIRFLOW_URL", "http://localhost:8080")

TERMINAL_PAGES = {
    "dvc_versions.png": ("DVC: versions, status and diff", "dvc_evidence.txt"),
    "kubernetes_healthy.png": ("kubectl: healthy deployment", "kubectl_healthy.txt"),
    "kubernetes_bad_image.png": ("kubectl: injected bad image", "kubectl_pods_bad_image.txt"),
    "kubernetes_bad_probe.png": ("kubectl: injected bad readiness probe", "kubectl_pods_bad_probe.txt"),
    "agent_scenario5_multi_tool.png": ("Agent: integration scenario", "scenario_5_multi_tool_reasoning.txt"),
    "agent_k8s_diagnosis_crashloop.png": ("Agent: CrashLoopBackOff diagnosis", "failure_k8s_crashloop.txt"),
    "agent_airflow_gate_failure.png": ("Agent: quality gate failure diagnosis", "failure_airflow_quality_gate.txt"),
    "agent_log_sample.png": ("Structured tool-call log (logs/agent.jsonl)", "log_sample.jsonl"),
}


def terminal_html(title: str, text: str) -> str:
    return f"""<html><body style="margin:0;background:#0d1117"><div style="font:600 13px Segoe UI;color:#8b949e;
    padding:10px 16px;border-bottom:1px solid #30363d">{html.escape(title)}</div>
    <pre style="margin:0;padding:16px;color:#e6edf3;font:12.5px/1.45 Consolas,monospace;white-space:pre-wrap;
    width:1180px">{html.escape(text)}</pre></body></html>"""


def main() -> None:
    SHOTS.mkdir(exist_ok=True)
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=MLFLOW)
    exp = client.get_experiment_by_name("training_pipeline")
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=10)
    passed = next(r for r in runs if r.data.tags.get("quality_gate") == "passed")

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge")
        page = browser.new_page(viewport={"width": 1400, "height": 850})

        def shot(name: str, url: str, wait_ms: int = 2500) -> None:
            page.goto(url)
            page.wait_for_timeout(wait_ms)
            page.screenshot(path=str(SHOTS / name))
            print("saved", name)

        page.goto(f"{MLFLOW}/#/experiments/{exp.experiment_id}/runs")
        page.wait_for_timeout(3000)
        close = page.get_by_role("button", name="Close")  # MLflow's assistant side panel
        if close.count():
            close.first.click()
        page.wait_for_timeout(2500)
        page.screenshot(path=str(SHOTS / "mlflow_experiment_runs.png"))
        print("saved mlflow_experiment_runs.png")
        shot("mlflow_run_provenance.png", f"{MLFLOW}/#/experiments/{exp.experiment_id}/runs/{passed.info.run_id}", 4000)
        shot("mlflow_model_registry.png", f"{MLFLOW}/#/models", 3000)

        page.goto(f"{AIRFLOW}/auth/login")
        page.fill("input[name=username]", os.environ["AIRFLOW_USERNAME"])
        page.fill("input[name=password]", os.environ["AIRFLOW_PASSWORD"])
        page.keyboard.press("Enter")
        page.wait_for_timeout(3000)
        shot("airflow_dag_overview.png", f"{AIRFLOW}/dags/training_pipeline", 4000)
        good = next(r for r in runs if r.data.tags.get("airflow_run_id") or r.data.params.get("airflow_run_id"))
        run_id = good.data.params.get("airflow_run_id", good.data.tags.get("airflow_run_id"))
        shot("airflow_dag_run_graph.png", f"{AIRFLOW}/dags/training_pipeline/runs/{run_id}", 4000)
        shot("airflow_dag_runs.png", f"{AIRFLOW}/dags/training_pipeline/runs", 3000)
        for arg in sys.argv[1:]:  # optional: failed run ids -> task page with logs
            shot(f"airflow_failed_{arg[-8:]}.png", f"{AIRFLOW}/dags/training_pipeline/runs/{arg}/tasks/quality_gate", 4000)

        for name, (title, src) in TERMINAL_PAGES.items():
            text = (EVID / src).read_text(encoding="utf-8")
            page.set_content(terminal_html(title, text[:6500]))
            page.set_viewport_size({"width": 1230, "height": 700})
            page.screenshot(path=str(SHOTS / name), full_page=True)
            print("saved", name)
        browser.close()


if __name__ == "__main__":
    main()
