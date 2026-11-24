import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from deepeval.confident.api import Endpoints, HttpMethods
from deepeval.test_run import github_actions
from deepeval.test_run.github_actions import (
    get_github_actions_git_context,
    link_test_run_to_github_repository,
)

HEAD_SHA = "0123456789abcdef0123456789abcdef01234567"
MERGE_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def _env(tmp_path: Path, event_name: str, payload: Dict[str, Any], **extra):
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(payload))
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": "acme/support-bot",
        "GITHUB_REPOSITORY_ID": "4242",
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_SHA": MERGE_SHA,
        "GITHUB_RUN_ID": "9001",
        "GITHUB_RUN_ATTEMPT": "2",
        **extra,
    }


PULL_REQUEST_PAYLOAD = {
    "pull_request": {
        "number": 17,
        "head": {"sha": HEAD_SHA},
        "base": {"ref": "main"},
    },
    "repository": {"id": 4242, "default_branch": "main"},
}


def test_pull_request_uses_the_head_commit_not_the_merge_commit(tmp_path):
    env = _env(tmp_path, "pull_request", PULL_REQUEST_PAYLOAD)

    assert get_github_actions_git_context(env) == {
        "repoOwner": "acme",
        "repoName": "support-bot",
        "repoId": 4242,
        "refType": "PULL_REQUEST",
        "prNumber": 17,
        "headSha": HEAD_SHA,
        "baseBranch": "main",
        "defaultBranch": "main",
        "workflowRunId": 9001,
        "workflowRunAttempt": 2,
    }


def test_push_to_a_branch_reports_the_branch_and_its_commit(tmp_path):
    env = _env(
        tmp_path,
        "push",
        {"repository": {"id": 4242, "default_branch": "main"}},
        GITHUB_REF_NAME="release",
        GITHUB_REF_TYPE="branch",
    )

    context = get_github_actions_git_context(env)

    assert context["refType"] == "BRANCH"
    assert context["baseBranch"] == "release"
    assert context["headSha"] == MERGE_SHA
    assert context["defaultBranch"] == "main"
    assert "prNumber" not in context


def test_schedule_without_a_repository_payload_runs_on_the_default_branch(
    tmp_path,
):
    env = _env(tmp_path, "schedule", {}, GITHUB_REF_NAME="main")

    context = get_github_actions_git_context(env)

    assert context["defaultBranch"] == "main"
    assert context["baseBranch"] == "main"


@pytest.mark.parametrize(
    "event_name, payload, extra",
    [
        (
            "push",
            {"repository": {"default_branch": "main"}},
            {"GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v1"},
        ),
        ("merge_group", {"repository": {"default_branch": "main"}}, {}),
        ("pull_request", {"pull_request": {"head": {"sha": HEAD_SHA}}}, {}),
        ("workflow_dispatch", {}, {"GITHUB_REF_NAME": "main"}),
    ],
    ids=[
        "tag push",
        "merge queue",
        "pull request without a number",
        "no default branch",
    ],
)
def test_events_without_a_branch_or_pull_request_are_not_reported(
    tmp_path, event_name, payload, extra
):
    assert (
        get_github_actions_git_context(
            _env(tmp_path, event_name, payload, **extra)
        )
        is None
    )


def test_outside_github_actions_nothing_is_reported(tmp_path):
    env = _env(tmp_path, "pull_request", PULL_REQUEST_PAYLOAD)
    env.pop("GITHUB_ACTIONS")

    assert get_github_actions_git_context(env) is None


class _FakeApi:
    calls: List[Dict[str, Any]] = []
    error: Exception = None

    def __init__(self, *args, **kwargs):
        pass

    def send_request(self, **kwargs):
        _FakeApi.calls.append(kwargs)
        if _FakeApi.error:
            raise _FakeApi.error
        return {"id": "gate-run-1"}, "https://app.confident-ai.com/x"


@pytest.fixture
def fake_api(monkeypatch, tmp_path):
    _FakeApi.calls = []
    _FakeApi.error = None
    monkeypatch.setattr(github_actions, "Api", _FakeApi)
    for name, value in _env(
        tmp_path, "pull_request", PULL_REQUEST_PAYLOAD
    ).items():
        monkeypatch.setenv(name, value)
    return _FakeApi


def test_link_posts_the_test_run_exit_code_and_git_context(fake_api, capsys):
    link_test_run_to_github_repository("test-run-1", 1)

    assert len(fake_api.calls) == 1
    call = fake_api.calls[0]
    assert call["method"] is HttpMethods.POST
    assert call["endpoint"] is Endpoints.EVAL_GATE_UNIT_TEST_RUNS_ENDPOINT
    assert call["body"]["testRunId"] == "test-run-1"
    assert call["body"]["exitCode"] == 1
    assert call["body"]["git"]["prNumber"] == 17
    assert "Linked this test run to acme/support-bot" in capsys.readouterr().out


def test_link_warns_instead_of_raising_when_the_server_rejects_it(
    fake_api, capsys
):
    fake_api.error = Exception("Not Found")

    link_test_run_to_github_repository("test-run-1", 0)

    assert "Warning: couldn't link this test run to acme/support-bot" in (
        capsys.readouterr().out
    )


def test_link_is_skipped_when_opted_out(fake_api, monkeypatch):
    monkeypatch.setenv("CONFIDENT_CI_LINK_OPT_OUT", "1")

    link_test_run_to_github_repository("test-run-1", 0)

    assert fake_api.calls == []
