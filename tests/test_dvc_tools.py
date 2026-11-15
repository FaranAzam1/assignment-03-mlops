"""DVC tools against a real throw-away Git repo holding hand-written .dvc files."""

import subprocess

import pytest

from agent.tools import mlflow_tools
from agent.tools.base import execute_tool
from tests.conftest import RUN_A, FakeMlflow, make_run

MD5_1, MD5_2 = "1" * 32, "2" * 32


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _dvc_file(repo, md5, size):
    (repo / "data").mkdir(exist_ok=True)
    (repo / "data" / "dataset.csv.dvc").write_text(
        f"outs:\n- md5: {md5}\n  size: {size}\n  hash: md5\n  path: dataset.csv\n"
    )


@pytest.fixture
def git_repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "tester")
    _dvc_file(tmp_path, MD5_1, 100)
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "data v1")
    _git(tmp_path, "tag", "data-v1")
    _dvc_file(tmp_path, MD5_2, 250)
    _git(tmp_path, "commit", "-qam", "data v2")
    _git(tmp_path, "tag", "data-v2")
    return tmp_path


def test_list_and_current_version(git_repo):
    listed = execute_tool("dvc_list_datasets", {}).data
    assert listed["count"] == 1 and listed["datasets"][0]["md5"] == MD5_2
    v = execute_tool("dvc_get_dataset_version", {}).data
    assert v["md5"] == MD5_2 and "data-v2" in v["last_changed_in"]["tags"]


def test_history_lists_both_versions(git_repo):
    h = execute_tool("dvc_list_dataset_history", {}).data["versions"]
    assert [x["md5"] for x in h] == [MD5_2, MD5_1]
    assert h[1]["tags"] == ["data-v1"]


def test_compare_versions(git_repo):
    d = execute_tool("dvc_compare_dataset_versions", {"rev_a": "data-v1", "rev_b": "data-v2"}).data
    assert not d["identical"] and d["size_delta_bytes"] == 150
    same = execute_tool("dvc_compare_dataset_versions", {"rev_a": "data-v2"}).data
    assert same["identical"]


def test_compare_unknown_revision_is_error(git_repo):
    res = execute_tool("dvc_compare_dataset_versions", {"rev_a": "no-such-tag"})
    assert res.status == "error" and "no-such-tag" in res.error


def test_reproduce_instructions_use_real_values(git_repo):
    d = execute_tool("dvc_explain_reproduce", {"rev": "data-v1"}).data
    assert d["expected_md5"] == MD5_1
    assert d["steps"][0] == "git checkout data-v1 -- data/dataset.csv.dvc"


def test_run_provenance_resolves_md5_to_commit_and_tag(git_repo):
    client = FakeMlflow([make_run(RUN_A, 0.9, md5=MD5_1)])
    mlflow_tools.set_client_factory(lambda: client)
    d = execute_tool("dvc_get_run_dataset_provenance", {"run_id": RUN_A}).data
    assert d["recorded_md5"] == MD5_1 and d["dataset_version"] == ["data-v1"]
    assert "git checkout" in d["reproduce_hint"]


def test_run_without_dvc_tag_has_unknown_provenance(git_repo):
    run = make_run(RUN_A, 0.9)
    del run.data.tags["dvc.md5"]
    client = FakeMlflow([run])
    mlflow_tools.set_client_factory(lambda: client)
    res = execute_tool("dvc_get_run_dataset_provenance", {"run_id": RUN_A})
    assert res.status == "error" and "unknown" in res.error


def test_untracked_dataset_reported(tmp_path):
    res = execute_tool("dvc_get_dataset_version", {"path": "data/missing.csv"})
    assert res.status == "error" and "not tracked" in res.error


def test_path_cannot_escape_repo(git_repo):
    res = execute_tool("dvc_get_dataset_version", {"path": "../../etc/passwd"})
    assert res.status == "error"
