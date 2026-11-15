"""Training pipeline logic shared by the Airflow DAG and the command line.

Stages (same as the DAG): validate -> preprocess -> train -> evaluate/log to MLflow -> gate.
Every stage is a plain function so it can be unit tested without Airflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent.parent))
DEFAULT_DATASET = "data/dataset.csv"
TARGET = "label"
MIN_ROWS = 200
REQUIRED_CLASSES = {0, 1}


class DataValidationError(Exception):
    """Dataset failed validation; retrying cannot fix it."""


class QualityGateError(Exception):
    """Model did not meet the quality threshold; it must not be promoted."""


# --------------------------------------------------------------------------- DVC helpers

def dvc_md5(path: Path) -> str:
    """md5 exactly as DVC 3 records it (`hash: md5` = raw file bytes, no line-ending changes)."""
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324 - matches DVC's content hash


def recorded_md5(dataset: str = DEFAULT_DATASET, root: Path = REPO_ROOT) -> str | None:
    f = root / (dataset + ".dvc")
    if not f.exists():
        return None
    doc = yaml.load(f.read_text(encoding="utf-8"), Loader=yaml.BaseLoader) or {}  # noqa: S506
    outs = doc.get("outs") or []
    return outs[0].get("md5") if outs else None


def git_commit(root: Path = REPO_ROOT) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=5, check=False)
        return out.stdout.strip() or os.environ.get("GIT_COMMIT", "unknown")
    except (OSError, subprocess.TimeoutExpired):
        return os.environ.get("GIT_COMMIT", "unknown")


# --------------------------------------------------------------------------- stages

def validate_dataset(dataset: str = DEFAULT_DATASET, root: Path = REPO_ROOT) -> dict[str, Any]:
    path = root / dataset
    if not path.exists():
        raise DataValidationError(f"dataset not found: {dataset} (run `dvc checkout`/`dvc pull`)")
    md5 = dvc_md5(path)
    expected = recorded_md5(dataset, root)
    if expected and md5 != expected:
        raise DataValidationError(
            f"dataset md5 {md5} does not match DVC-recorded {expected}: data changed without `dvc add`")
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise DataValidationError(f"missing target column '{TARGET}'")
    if len(df) < MIN_ROWS:
        raise DataValidationError(f"only {len(df)} rows, need at least {MIN_ROWS}")
    if df.drop(columns=[TARGET]).isna().any().any():
        raise DataValidationError("dataset contains missing feature values")
    if set(df[TARGET].unique()) != REQUIRED_CLASSES:
        raise DataValidationError(f"target classes {sorted(df[TARGET].unique())}, expected [0, 1]")
    return {"dataset": dataset, "md5": md5, "rows": int(len(df)), "features": int(df.shape[1] - 1),
            "matches_dvc": expected == md5 if expected else None}


def preprocess(dataset: str, workdir: Path, root: Path = REPO_ROOT, seed: int = 42) -> dict[str, str]:
    df = pd.read_csv(root / dataset)
    X, y = df.drop(columns=[TARGET]).to_numpy(), df[TARGET].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    workdir.mkdir(parents=True, exist_ok=True)
    np.savez(workdir / "split.npz", X_tr=scaler.transform(X_tr), X_te=scaler.transform(X_te),
             y_tr=y_tr, y_te=y_te)
    joblib.dump(scaler, workdir / "scaler.joblib")
    return {"split": str(workdir / "split.npz"), "scaler": str(workdir / "scaler.joblib")}


def train_model(workdir: Path, model_type: str = "random_forest", seed: int = 42) -> str:
    d = np.load(workdir / "split.npz")
    model = (RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed)
             if model_type == "random_forest" else LogisticRegression(max_iter=500, random_state=seed))
    model.fit(d["X_tr"], d["y_tr"])
    path = workdir / "model.joblib"
    joblib.dump(model, path)
    return str(path)


def evaluate(workdir: Path) -> dict[str, float]:
    d = np.load(workdir / "split.npz")
    model = joblib.load(workdir / "model.joblib")
    pred, proba = model.predict(d["X_te"]), model.predict_proba(d["X_te"])[:, 1]
    return {
        "accuracy": float(accuracy_score(d["y_te"], pred)),
        "precision": float(precision_score(d["y_te"], pred)),
        "recall": float(recall_score(d["y_te"], pred)),
        "f1": float(f1_score(d["y_te"], pred)),
        "roc_auc": float(roc_auc_score(d["y_te"], proba)),
    }


def log_to_mlflow(metrics: dict[str, float], data_info: dict[str, Any], workdir: Path,
                  model_type: str, tracking_uri: str, experiment: str, run_name: str | None = None,
                  extra_params: dict[str, Any] | None = None, root: Path = REPO_ROOT) -> str:
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"model_type": model_type, **(extra_params or {})})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({
            "dvc.path": data_info["dataset"], "dvc.md5": data_info["md5"],
            "dvc.rows": str(data_info["rows"]), "git.commit": git_commit(root),
            "quality_gate": "pending",
        })
        mlflow.log_artifact(str(workdir / "model.joblib"))
        mlflow.log_artifact(str(workdir / "scaler.joblib"))
        info = mlflow.sklearn.log_model(
            joblib.load(workdir / "model.joblib"), name="model", serialization_format="cloudpickle")
        mlflow.set_tag("model_uri", info.model_uri)
        return run.info.run_id


def quality_gate(run_id: str, metrics: dict[str, float], threshold: float, tracking_uri: str,
                 metric: str = "f1", register_as: str | None = "ml-inference-model") -> dict[str, Any]:
    """Pass: tag the run, register the model version. Fail: tag the run and raise."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    value = metrics[metric]
    if value < threshold:
        client.set_tag(run_id, "quality_gate", "failed")
        raise QualityGateError(f"{metric}={value:.4f} below threshold {threshold:.4f}; model not promoted")
    client.set_tag(run_id, "quality_gate", "passed")
    version = None
    if register_as:
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = client.get_run(run_id).data.tags["model_uri"]
        mv = mlflow.register_model(model_uri, register_as)
        client.set_registered_model_alias(register_as, "candidate", mv.version)
        version = mv.version
    return {"passed": True, "metric": metric, "value": value, "threshold": threshold,
            "registered_version": version}


def export_model(workdir: Path, out_dir: Path) -> Path:
    """Publish the gated model (scaler + model in one bundle) for the inference image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {"scaler": joblib.load(workdir / "scaler.joblib"), "model": joblib.load(workdir / "model.joblib")}
    target = out_dir / "model.joblib"
    joblib.dump(bundle, target)
    return target


def run_pipeline(tracking_uri: str, experiment: str, threshold: float = 0.80,
                 model_type: str = "random_forest", dataset: str = DEFAULT_DATASET,
                 workdir: Path | None = None, root: Path = REPO_ROOT, export: bool = True) -> dict[str, Any]:
    workdir = workdir or root / "airflow-work" / "cli"
    info = validate_dataset(dataset, root)
    preprocess(dataset, workdir, root)
    train_model(workdir, model_type)
    metrics = evaluate(workdir)
    run_id = log_to_mlflow(metrics, info, workdir, model_type, tracking_uri, experiment, root=root)
    gate = quality_gate(run_id, metrics, threshold, tracking_uri)
    if export:
        export_model(workdir, root / "models")
    return {"run_id": run_id, "metrics": metrics, "gate": gate, "data": info}


def main() -> None:
    p = argparse.ArgumentParser(description="Run the training pipeline without Airflow")
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    p.add_argument("--experiment", default="training_pipeline")
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--model", default="random_forest", choices=["random_forest", "logistic_regression"])
    a = p.parse_args()
    print(json.dumps(run_pipeline(a.tracking_uri, a.experiment, a.threshold, a.model), indent=2))


if __name__ == "__main__":
    main()
