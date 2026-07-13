import json
import os
from typing import Optional

import requests
from deepeval.confident.api import get_base_api_url

PR_COMMENTS_ENDPOINT = "/v1/github-app/test-run/comments"
_OIDC_AUDIENCE = "confident-deepeval"
_APP = "deepeval"


def oidc_request_available() -> bool:
    return bool(
        os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
        and os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    )


def _github_oidc_token() -> Optional[str]:
    req_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    req_url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    if not req_token or not req_url:
        return None
    resp = requests.get(
        f"{req_url}&audience={_OIDC_AUDIENCE}",
        headers={"Authorization": f"bearer {req_token}"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("value")


def _repo() -> Optional[str]:
    return os.environ.get("GITHUB_REPOSITORY")


def _pr_number() -> Optional[int]:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        try:
            with open(event_path) as f:
                event = json.load(f)
            num = (event.get("pull_request") or {}).get("number")
            if num is None:
                num = (event.get("inputs") or {}).get("pr_number")
            if num is not None:
                return int(num)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    parts = os.environ.get("GITHUB_REF", "").split("/")
    if len(parts) >= 3 and parts[1] == "pull":
        try:
            return int(parts[2])
        except ValueError:
            return None
    return None


def post_pr_comment_via_bot(
    markdown: str,
    repo: Optional[str] = None,
    pr_number: Optional[int] = None,
    api_url: Optional[str] = None,
) -> bool:
    repo = repo or _repo()
    if pr_number is None:
        pr_number = _pr_number()
    oidc = _github_oidc_token()

    if not repo or pr_number is None:
        raise RuntimeError(
            "Could not resolve the repository or PR number. PR comments are "
            "only posted from a GitHub Actions pull_request job."
        )
    if not oidc:
        raise RuntimeError(
            "Could not obtain a GitHub OIDC token. Grant the workflow "
            "`permissions: id-token: write`."
        )

    base = api_url or get_base_api_url()
    payload = {
        "app": _APP,
        "repo": repo,
        "pr": pr_number,
        "oidc": oidc,
        "body": markdown,
    }
    resp = requests.post(
        f"{base}{PR_COMMENTS_ENDPOINT}", json=payload, timeout=30
    )
    resp.raise_for_status()
    return True
