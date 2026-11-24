"""Link a `deepeval test run` from GitHub Actions to its repository on
Confident AI, so the run is listed on the CI/CD page with its pull request,
commit and branch."""

import json
import os
from typing import Any, Dict, Mapping, Optional

from deepeval.config.settings import get_settings
from deepeval.confident.api import Api, Endpoints, HttpMethods

PULL_REQUEST_EVENTS = frozenset({"pull_request", "pull_request_target"})
BRANCH_EVENTS = frozenset({"push", "workflow_dispatch", "schedule"})


def _read_event_payload(env: Mapping[str, str]) -> Dict[str, Any]:
    path = env.get("GITHUB_EVENT_PATH")
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _as_positive_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def get_github_actions_git_context(
    env: Mapping[str, str] = os.environ,
) -> Optional[Dict[str, Any]]:
    """The repository, commit and pull request of the current GitHub Actions
    run, shaped for Confident AI, or None outside GitHub Actions or for an
    event that has no branch or pull request (a tag push, a merge queue)."""
    if env.get("GITHUB_ACTIONS") != "true":
        return None

    owner, _, name = env.get("GITHUB_REPOSITORY", "").partition("/")
    event_name = env.get("GITHUB_EVENT_NAME")
    payload = _read_event_payload(env)
    repository = payload.get("repository") or {}
    repo_id = _as_positive_int(
        env.get("GITHUB_REPOSITORY_ID") or repository.get("id")
    )

    pr_number: Optional[int] = None
    if event_name in PULL_REQUEST_EVENTS:
        pull_request = payload.get("pull_request") or {}
        ref_type = "PULL_REQUEST"
        pr_number = _as_positive_int(pull_request.get("number"))
        # GITHUB_SHA is GitHub's temporary merge commit on pull request
        # events; the pull request's own head commit is in the payload.
        head_sha = (pull_request.get("head") or {}).get("sha")
        base_branch = (pull_request.get("base") or {}).get("ref") or env.get(
            "GITHUB_BASE_REF"
        )
        if pr_number is None:
            return None
    elif (
        event_name in BRANCH_EVENTS
        and env.get("GITHUB_REF_TYPE", "branch") == "branch"
    ):
        ref_type = "BRANCH"
        head_sha = env.get("GITHUB_SHA")
        base_branch = env.get("GITHUB_REF_NAME")
    else:
        return None

    # Scheduled runs always run on the default branch, and their payload
    # doesn't carry the repository.
    default_branch = repository.get("default_branch") or (
        base_branch if event_name == "schedule" else None
    )
    if not all([owner, name, repo_id, head_sha, base_branch, default_branch]):
        return None

    context: Dict[str, Any] = {
        "repoOwner": owner,
        "repoName": name,
        "repoId": repo_id,
        "refType": ref_type,
        "headSha": head_sha,
        "baseBranch": base_branch,
        "defaultBranch": default_branch,
    }
    if pr_number is not None:
        context["prNumber"] = pr_number
    workflow_run_id = _as_positive_int(env.get("GITHUB_RUN_ID"))
    workflow_run_attempt = _as_positive_int(env.get("GITHUB_RUN_ATTEMPT"))
    if workflow_run_id is not None:
        context["workflowRunId"] = workflow_run_id
    if workflow_run_attempt is not None:
        context["workflowRunAttempt"] = workflow_run_attempt
    return context


def link_test_run_to_github_repository(
    test_run_id: str, exit_code: int
) -> None:
    """Send the GitHub Actions context of an uploaded test run to Confident
    AI. Never raises: a failure here must not change the run's exit code, and
    an older Confident AI server answers this call with an error."""
    if get_settings().CONFIDENT_CI_LINK_OPT_OUT:
        return
    git = get_github_actions_git_context()
    if git is None:
        return

    repository = f"{git['repoOwner']}/{git['repoName']}"
    try:
        Api().send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.EVAL_GATE_UNIT_TEST_RUNS_ENDPOINT,
            body={"testRunId": test_run_id, "exitCode": exit_code, "git": git},
        )
    except Exception as e:
        print(
            f"Warning: couldn't link this test run to {repository} on "
            f"Confident AI's CI/CD page: {e}"
        )
        return
    print(
        f"✓ Linked this test run to {repository} on Confident AI's CI/CD page."
    )
