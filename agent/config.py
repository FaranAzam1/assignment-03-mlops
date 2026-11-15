"""Runtime configuration. Everything comes from environment variables (or a local .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


@dataclass(frozen=True)
class Settings:
    repo_root: Path = REPO_ROOT_DEFAULT
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    llm_timeout_s: float = 120.0
    mlflow_tracking_uri: str = "http://localhost:5000"
    airflow_url: str = "http://localhost:8080"
    airflow_username: str = "admin"
    airflow_password: str = field(default="", repr=False)
    allowed_dag_ids: tuple[str, ...] = ("training_pipeline",)
    k8s_namespace: str = "mlops"
    allowed_namespaces: tuple[str, ...] = ("mlops",)
    k8s_deployment: str = "ml-inference"
    tool_timeout_s: float = 20.0
    log_file: Path = REPO_ROOT_DEFAULT / "logs" / "agent.jsonl"
    dataset_path: str = "data/dataset.csv"
    default_experiment: str = "training_pipeline"
    primary_metric: str = "f1"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(REPO_ROOT_DEFAULT / ".env", override=False)
        e = os.environ
        repo_root = Path(e.get("REPO_ROOT", str(REPO_ROOT_DEFAULT)))
        return cls(
            repo_root=repo_root,
            ollama_url=e.get("OLLAMA_URL", cls.ollama_url).rstrip("/"),
            ollama_model=e.get("OLLAMA_MODEL", cls.ollama_model),
            llm_timeout_s=float(e.get("LLM_TIMEOUT_S", cls.llm_timeout_s)),
            mlflow_tracking_uri=e.get("MLFLOW_TRACKING_URI", cls.mlflow_tracking_uri),
            airflow_url=e.get("AIRFLOW_URL", cls.airflow_url).rstrip("/"),
            airflow_username=e.get("AIRFLOW_USERNAME", cls.airflow_username),
            airflow_password=e.get("AIRFLOW_PASSWORD", ""),
            allowed_dag_ids=_csv(e.get("ALLOWED_DAG_IDS", "training_pipeline")),
            k8s_namespace=e.get("K8S_NAMESPACE", cls.k8s_namespace),
            allowed_namespaces=_csv(e.get("ALLOWED_NAMESPACES", "mlops")),
            k8s_deployment=e.get("K8S_DEPLOYMENT", cls.k8s_deployment),
            tool_timeout_s=float(e.get("TOOL_TIMEOUT_S", cls.tool_timeout_s)),
            log_file=Path(e.get("AGENT_LOG_FILE", str(repo_root / "logs" / "agent.jsonl"))),
            dataset_path=e.get("DATASET_PATH", cls.dataset_path),
            default_experiment=e.get("MLFLOW_EXPERIMENT", cls.default_experiment),
            primary_metric=e.get("PRIMARY_METRIC", cls.primary_metric),
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def set_settings(settings: Settings | None) -> None:
    """Override (tests) or reset (None) the cached settings."""
    global _settings
    _settings = settings
