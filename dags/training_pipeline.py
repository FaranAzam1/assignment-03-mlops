"""Airflow 3 training DAG: validate -> preprocess -> train -> evaluate/log -> quality gate.

* retries: every task retries twice, except validation and the quality gate, which raise
  AirflowFailException (a retry cannot fix bad data or a bad model).
* dependencies: strictly linear; a failed validation or gate stops everything downstream,
  and only a passed gate registers a model / exports it for deployment.
* params (set at trigger time via `conf`):
    f1_threshold      quality gate threshold (default 0.80; 0.99 forces a gate failure)
    model_type        random_forest | logistic_regression
    inject_failure    none | validation | training   (deliberate failure injection for demos)
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path

from airflow.exceptions import AirflowFailException
from airflow.sdk import DAG, Param, get_current_context, task

from training import train as T

log = logging.getLogger("training_pipeline")

REPO = Path(os.environ.get("REPO_ROOT", "/opt/airflow/repo"))
WORK = Path(os.environ.get("WORK_ROOT", "/opt/airflow/work"))
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "training_pipeline")

default_args = {"retries": 2, "retry_delay": timedelta(seconds=10), "owner": "mlops"}


def _p(name: str):
    return get_current_context()["params"][name]


def _workdir() -> Path:
    return WORK / get_current_context()["run_id"].replace(":", "_").replace("+", "_")


with DAG(
    dag_id="training_pipeline",
    description="Validate, preprocess, train, evaluate, gate",
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["mlops", "training"],
    params={
        "f1_threshold": Param(0.80, type="number", minimum=0, maximum=1),
        "model_type": Param("random_forest", enum=["random_forest", "logistic_regression"]),
        "inject_failure": Param("none", enum=["none", "validation", "training"]),
        "dataset": Param("data/dataset.csv", type="string"),
    },
) as dag:

    @task(retries=0)
    def validate_dataset() -> dict:
        if _p("inject_failure") == "validation":
            raise AirflowFailException("injected failure: dataset validation (deliberate)")
        try:
            info = T.validate_dataset(_p("dataset"), REPO)
        except T.DataValidationError as exc:
            log.error("validation failed: %s", exc)
            raise AirflowFailException(str(exc)) from exc
        log.info("dataset valid: %s", info)
        return info

    @task
    def preprocess_data(info: dict) -> dict:
        out = T.preprocess(info["dataset"], _workdir(), REPO)
        log.info("preprocessed into %s", out)
        return out

    @task
    def train_model(_prep: dict) -> str:
        if _p("inject_failure") == "training":
            raise RuntimeError("injected failure: training crashed (deliberate)")
        path = T.train_model(_workdir(), _p("model_type"))
        log.info("model saved to %s", path)
        return path

    @task
    def evaluate_and_log(info: dict, _model: str) -> dict:
        metrics = T.evaluate(_workdir())
        run_id = T.log_to_mlflow(
            metrics, info, _workdir(), _p("model_type"), TRACKING_URI, EXPERIMENT,
            run_name=get_current_context()["run_id"], root=REPO,
            extra_params={"f1_threshold": _p("f1_threshold"), "airflow_run_id": get_current_context()["run_id"]},
        )
        log.info("logged MLflow run %s metrics=%s", run_id, metrics)
        return {"run_id": run_id, "metrics": metrics}

    @task(retries=0)
    def quality_gate(result: dict) -> dict:
        try:
            gate = T.quality_gate(result["run_id"], result["metrics"], float(_p("f1_threshold")), TRACKING_URI)
        except T.QualityGateError as exc:
            log.error("quality gate failed: %s", exc)
            raise AirflowFailException(str(exc)) from exc
        T.export_model(_workdir(), REPO / "models")  # downstream promotion only after the gate
        log.info("quality gate passed: %s", gate)
        return gate

    info_ = validate_dataset()
    prep_ = preprocess_data(info_)
    model_ = train_model(prep_)
    result_ = evaluate_and_log(info_, model_)
    quality_gate(result_)
