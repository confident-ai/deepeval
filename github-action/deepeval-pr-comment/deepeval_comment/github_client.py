"""Post a Markdown comment to a GitHub pull request via the REST API."""

from __future__ import annotations

import os

import requests


def post_pr_comment(
    repo: str,
    pr_number: str,
    body: str,
    token: str,
    api_url: str = "https://api.github.com",
) -> int:
    """Create an issue comment on the given PR.

    Returns the HTTP status code from the GitHub API.
    """
    url = f"{api_url}/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.post(
        url, json={"body": body}, headers=headers, timeout=30
    )
    response.raise_for_status()
    return response.status_code


def extract_pr_number(github_ref: str) -> str | None:
    """Extract the PR number from a ``refs/pull/<n>/merge`` GitHub ref."""
    if not github_ref:
        return None
    parts = github_ref.split("/")
    # refs/pull/123/merge -> ['refs', 'pull', '123', 'merge']
    if len(parts) >= 3 and parts[1] == "pull":
        return parts[2]
    return None
