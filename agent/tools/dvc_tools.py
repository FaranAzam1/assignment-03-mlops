"""DVC tools: dataset discovery, versions, comparison, change detection, provenance.

Read-only. Git/DVC are invoked through an allow-listed subprocess wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from agent.config import get_settings
from agent.tools.base import ToolError, params, run_command, tool

_PATH = {"type": "string", "description": "Dataset path relative to the repo, e.g. data/dataset.csv"}


def _root() -> Path:
    return get_settings().repo_root


def _budget() -> float:
    """Subprocess time budget: must finish inside the tool deadline."""
    return get_settings().tool_timeout_s * 0.6


def _git(*args: str) -> str:
    return run_command(["git", *args], _root())


def _dvc_file(path: str) -> Path:
    rel = path if path.endswith(".dvc") else path + ".dvc"
    p = (_root() / rel).resolve()
    if _root().resolve() not in p.parents:
        raise ToolError("path escapes the repository")
    return p


def _read_dvc_text(text: str) -> dict[str, Any]:
    # BaseLoader keeps every scalar a string: an all-digit md5 must not become an int/float.
    doc = yaml.load(text, Loader=yaml.BaseLoader) or {}  # noqa: S506 - BaseLoader is safe
    outs = doc.get("outs") or []
    if not outs:
        raise ToolError("no outs entry in .dvc file")
    out = outs[0]
    size = out.get("size")
    return {
        "path": out.get("path"),
        "md5": out.get("md5"),
        "size": int(size) if size and size.isdigit() else None,
        "nfiles": out.get("nfiles"),
    }


def _md5_at(rev: str, path: str) -> dict[str, Any]:
    rel = (path if path.endswith(".dvc") else path + ".dvc").replace("\\", "/")
    try:
        text = _git("show", f"{rev}:{rel}")
    except ToolError as exc:
        raise ToolError(f"cannot read {rel} at revision '{rev}': {exc}") from exc
    return _read_dvc_text(text)


def _commit_for_md5(md5: str, path: str) -> dict[str, str] | None:
    """Find the Git commit that introduced a given DVC md5 (provenance by content hash)."""
    rel = (path if path.endswith(".dvc") else path + ".dvc").replace("\\", "/")
    out = _git("log", "--all", f"-S{md5}", "--format=%H|%h|%cI|%s", "--", rel).strip()
    if not out:
        return None
    full, short, when, subject = out.splitlines()[-1].split("|", 3)  # oldest commit
    tags = _git("tag", "--points-at", full).split()
    return {"commit": full, "short": short, "date": when, "subject": subject, "tags": tags}


@tool(
    "dvc_list_datasets",
    "dvc",
    "List datasets tracked by DVC (.dvc files and dvc.yaml outputs) with their md5 and size.",
)
def dvc_list_datasets() -> dict[str, Any]:
    root = _root()
    datasets = []
    for f in sorted(root.rglob("*.dvc")):
        if ".venv" in f.parts or ".dvc" in f.parts[:-1]:
            continue
        try:
            info = _read_dvc_text(f.read_text(encoding="utf-8"))
        except (ToolError, yaml.YAMLError, OSError):
            continue
        info["dvc_file"] = f.relative_to(root).as_posix()
        datasets.append(info)
    return {"count": len(datasets), "datasets": datasets}


@tool(
    "dvc_get_dataset_version",
    "dvc",
    "Get the current DVC version of a dataset: md5, size, last Git commit that changed it, "
    "and Git tags (data-vN) on that commit.",
    params({"path": _PATH}),
)
def dvc_get_dataset_version(path: str | None = None) -> dict[str, Any]:
    path = path or get_settings().dataset_path
    f = _dvc_file(path)
    if not f.exists():
        raise ToolError(f"{f.name} not found: dataset is not tracked by DVC")
    info = _read_dvc_text(f.read_text(encoding="utf-8"))
    rel = f.relative_to(_root()).as_posix()
    log = _git("log", "-1", "--format=%H|%h|%cI|%s", "--", rel).strip()
    commit = None
    if log:
        full, short, when, subject = log.split("|", 3)
        commit = {
            "commit": full,
            "short": short,
            "date": when,
            "subject": subject,
            "tags": _git("tag", "--points-at", full).split(),
        }
    head = _git("rev-parse", "--short", "HEAD").strip()
    return {**info, "dvc_file": rel, "last_changed_in": commit, "git_head": head}


@tool(
    "dvc_list_dataset_history",
    "dvc",
    "List every Git commit that changed a dataset's .dvc file, with the md5 recorded at each.",
    params({"path": _PATH, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}),
)
def dvc_list_dataset_history(path: str | None = None, limit: int = 10) -> dict[str, Any]:
    path = path or get_settings().dataset_path
    rel = _dvc_file(path).relative_to(_root()).as_posix()
    out = _git("log", f"-{limit}", "--format=%H|%h|%cI|%s", "--", rel).strip()
    history = []
    for line in out.splitlines():
        full, short, when, subject = line.split("|", 3)
        try:
            md5 = _md5_at(full, path)["md5"]
        except ToolError:
            md5 = None
        history.append(
            {
                "commit": short,
                "date": when,
                "subject": subject,
                "md5": md5,
                "tags": _git("tag", "--points-at", full).split(),
            }
        )
    return {"path": path, "versions": history}


@tool(
    "dvc_compare_dataset_versions",
    "dvc",
    "Compare a dataset between two Git revisions (commit, tag such as data-v1, or HEAD).",
    params(
        {
            "path": _PATH,
            "rev_a": {"type": "string", "description": "Older revision or tag"},
            "rev_b": {"type": "string", "description": "Newer revision or tag, default HEAD"},
        },
        ["rev_a"],
    ),
)
def dvc_compare_dataset_versions(
    rev_a: str, rev_b: str = "HEAD", path: str | None = None
) -> dict[str, Any]:
    path = path or get_settings().dataset_path
    a, b = _md5_at(rev_a, path), _md5_at(rev_b, path)
    result: dict[str, Any] = {
        "path": path,
        "rev_a": {"rev": rev_a, **a},
        "rev_b": {"rev": rev_b, **b},
        "identical": a["md5"] == b["md5"],
        "size_delta_bytes": (b.get("size") or 0) - (a.get("size") or 0),
    }
    try:
        result["dvc_diff"] = json.loads(
            run_command(
                ["dvc", "diff", "--json", rev_a, rev_b, "--targets", path],
                _root(),
                timeout=_budget(),
            )
        )
    except ToolError as exc:  # dvc missing or no cache: md5 comparison above still holds
        result["dvc_diff_note"] = f"dvc diff unavailable: {exc}"
    return result


@tool(
    "dvc_check_data_changed",
    "dvc",
    "Detect whether DVC-tracked data in the workspace differs from what its .dvc file records.",
    params({"path": _PATH}),
)
def dvc_check_data_changed(path: str | None = None) -> dict[str, Any]:
    path = path or get_settings().dataset_path
    f = _dvc_file(path)
    if not f.exists():
        raise ToolError(f"{f.name} not found: dataset is not tracked by DVC")
    raw = run_command(["dvc", "status", "--json", f.relative_to(_root()).as_posix()], _root(), timeout=_budget())
    status = json.loads(raw or "{}")
    changed = bool(status)
    return {
        "path": path,
        "changed": changed,
        "dvc_status": status,
        "summary": "workspace data differs from the committed DVC version"
        if changed
        else "workspace data matches the committed DVC version",
    }


@tool(
    "dvc_explain_reproduce",
    "dvc",
    "Explain exactly how to restore a previous dataset version (Git revision or tag).",
    params({"path": _PATH, "rev": {"type": "string", "description": "Commit or tag"}}, ["rev"]),
)
def dvc_explain_reproduce(rev: str, path: str | None = None) -> dict[str, Any]:
    path = path or get_settings().dataset_path
    info = _md5_at(rev, path)
    try:
        remotes = run_command(["dvc", "remote", "list"], _root(), timeout=_budget()).strip()
    except ToolError:
        remotes = ""
    rel = path + ".dvc"
    steps = [
        f"git checkout {rev} -- {rel}",
        f"dvc checkout {path}   # from local cache, or: dvc pull {rel} to fetch from the remote",
        f"dvc status {rel}   # must report 'Data and pipelines are up to date'",
    ]
    return {
        "path": path,
        "rev": rev,
        "expected_md5": info["md5"],
        "remote_configured": remotes or None,
        "steps": steps,
        "restore_to_head": [f"git checkout HEAD -- {rel}", f"dvc checkout {path}"],
    }


@tool(
    "dvc_get_run_dataset_provenance",
    "dvc",
    "Read the DVC dataset reference recorded on an MLflow run (tags) and resolve it to the Git "
    "commit and tag that produced that exact data version. Answers 'which dataset version "
    "was used for run X'.",
    params({"run_id": {"type": "string", "description": "MLflow run id"}}, ["run_id"]),
)
def dvc_get_run_dataset_provenance(run_id: str) -> dict[str, Any]:
    from agent.tools.mlflow_tools import get_client

    run = get_client().get_run(run_id)
    tags = run.data.tags
    md5 = tags.get("dvc.md5")
    if not md5:
        raise ToolError(f"run {run_id} has no recorded dvc.md5 tag: provenance is unknown")
    dataset = tags.get("dvc.path", get_settings().dataset_path)
    commit = _commit_for_md5(md5, dataset)
    return {
        "run_id": run_id,
        "dataset": dataset,
        "recorded_md5": md5,
        "recorded_git_commit": tags.get("git.commit"),
        "resolved_commit": commit,
        "dataset_version": (commit or {}).get("tags") or None,
        "rows": tags.get("dvc.rows"),
        "reproduce_hint": (
            f"git checkout {commit['short']} -- {dataset}.dvc && dvc checkout {dataset}"
            if commit
            else "no Git commit contains this md5; the data may have been changed locally"
        ),
    }
