"""Training pipeline stages, including a real end-to-end run against a throw-away MLflow store."""

import hashlib

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification

from training import train as T


def _write_dataset(root, n=400, name="dataset.csv"):
    X, y = make_classification(n_samples=n, n_features=6, n_informative=4, random_state=1)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    df["label"] = y
    (root / "data").mkdir(exist_ok=True)
    df.to_csv(root / "data" / name, index=False)
    return df


def test_validation_accepts_good_data_and_reports_md5(tmp_path):
    _write_dataset(tmp_path)
    info = T.validate_dataset("data/dataset.csv", tmp_path)
    assert info["rows"] == 400 and info["features"] == 6 and len(info["md5"]) == 32


def test_validation_rejects_bad_data(tmp_path):
    with pytest.raises(T.DataValidationError, match="not found"):
        T.validate_dataset("data/dataset.csv", tmp_path)
    _write_dataset(tmp_path, n=50)
    with pytest.raises(T.DataValidationError, match="rows"):
        T.validate_dataset("data/dataset.csv", tmp_path)
    df = _write_dataset(tmp_path)
    df.loc[0, "f0"] = None
    df.to_csv(tmp_path / "data" / "dataset.csv", index=False)
    with pytest.raises(T.DataValidationError, match="missing"):
        T.validate_dataset("data/dataset.csv", tmp_path)


def test_validation_detects_data_changed_without_dvc_add(tmp_path):
    _write_dataset(tmp_path)
    (tmp_path / "data" / "dataset.csv.dvc").write_text("outs:\n- md5: " + "0" * 32 + "\n  path: dataset.csv\n")
    with pytest.raises(T.DataValidationError, match="does not match DVC"):
        T.validate_dataset("data/dataset.csv", tmp_path)


def test_md5_matches_dvc_hash_of_raw_bytes(tmp_path):
    f = tmp_path / "a.csv"
    raw = b"x,y\r\n1,2\r\n"  # CRLF as written by pandas on Windows
    f.write_bytes(raw)
    assert T.dvc_md5(f) == hashlib.md5(raw).hexdigest()


@pytest.fixture
def tracking(tmp_path):
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_pipeline_passes_gate_registers_model_and_records_provenance(tmp_path, tracking):
    _write_dataset(tmp_path)
    out = T.run_pipeline(tracking, "t", threshold=0.5, root=tmp_path, workdir=tmp_path / "w")
    client = MlflowClient(tracking_uri=tracking)
    run = client.get_run(out["run_id"])
    assert run.data.tags["quality_gate"] == "passed"
    assert run.data.tags["dvc.md5"] == out["data"]["md5"]
    assert run.data.metrics["f1"] > 0.5
    assert out["gate"]["registered_version"] is not None
    assert (tmp_path / "models" / "model.joblib").exists()


def test_failed_quality_gate_blocks_promotion(tmp_path, tracking):
    _write_dataset(tmp_path)
    with pytest.raises(T.QualityGateError, match="below threshold"):
        T.run_pipeline(tracking, "t", threshold=0.9999, root=tmp_path, workdir=tmp_path / "w")
    client = MlflowClient(tracking_uri=tracking)
    runs = client.search_runs([client.get_experiment_by_name("t").experiment_id])
    assert runs[0].data.tags["quality_gate"] == "failed"
    assert not client.search_registered_models()          # nothing registered
    assert not (tmp_path / "models" / "model.joblib").exists()  # nothing exported for deployment
