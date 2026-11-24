import json
import os
from typing import List, Optional

import requests
from deepeval.confident.api import get_base_api_url

from ..schema import CodeFinding

PR_COMMENTS_ENDPOINT = "/v1/github-app/code-scan/comments"
_OIDC_AUDIENCE = "confident-code-scan"
_APP = "deepteam"


def _github_oidc_token() -> Optional[str]:
    """
    Fetch the GitHub Actions OIDC token.
    """
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


def _pr_number() -> Optional[int]:
    """
    Resolve the PR number from the Actions environment.
    """
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


def post_pr_comments(
    findings: List[CodeFinding],
    repo: Optional[str] = None,
    pr_number: Optional[int] = None,
    api_url: Optional[str] = None,
) -> None:
    """
    Send findings to Confident AI, the platform posts them as deepteam[bot].
    """
    repo = repo or os.environ.get("GITHUB_REPOSITORY")
    if pr_number is None:
        pr_number = _pr_number()
    oidc = _github_oidc_token()

    if not repo or pr_number is None:
        raise RuntimeError(
            "`deepteam scan --comment` must run in a GitHub Actions "
            "pull_request job (could not resolve repository or PR number)."
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
        "findings": [
            finding.model_dump(exclude_none=True) 
            for finding in findings
        ],
    }
    resp = requests.post(
        f"{base}{PR_COMMENTS_ENDPOINT}", json=payload, timeout=30
    )
    resp.raise_for_status()
