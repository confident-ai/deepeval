"""Orchestrate a DeepEval run and post the result as a PR comment.

Inputs are read from ``INPUT_*`` environment variables (set by the composite
GitHub Action) with sane fallbacks to the standard ``GITHUB_*`` variables.
"""

from __future__ import annotations

import os
import subprocess
import sys

from deepeval_comment.formatter import format_summary, load_test_run
from deepeval_comment.github_client import extract_pr_number, post_pr_comment


def _find_run_json() -> str | None:
    hidden = os.getenv("DEEPEVAL_CACHE_FOLDER", ".deepeval")
    candidates = [
        os.path.join(hidden, ".latest_run_full.json"),
        os.path.join(hidden, ".temp_test_run_data.json"),
        os.path.join(hidden, ".latest_test_run.json"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def main() -> int:
    token = os.environ.get("INPUT_GITHUB_TOKEN") or os.environ.get(
        "GITHUB_TOKEN"
    )
    repo = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("INPUT_PR_NUMBER") or extract_pr_number(
        os.environ.get("GITHUB_REF", "")
    )
    command = os.environ.get("INPUT_TEST_COMMAND", "deepeval test run .")
    summary_only = (
        os.environ.get("INPUT_SUMMARY_ONLY", "false").lower() == "true"
    )
    fail_on_failure = (
        os.environ.get("INPUT_FAIL_ON_FAILURE", "false").lower() == "true"
    )

    if not token:
        print("deepeval-pr-comment: missing github_token", file=sys.stderr)
        return 2
    if not repo:
        print("deepeval-pr-comment: missing GITHUB_REPOSITORY", file=sys.stderr)
        return 2
    if not pr_number:
        print(
            "deepeval-pr-comment: could not determine PR number "
            "(set pr_number or run from a pull_request event)",
            file=sys.stderr,
        )
        return 2

    print(f"deepeval-pr-comment: running '{command}'")
    subprocess.run(command, shell=True, check=False)

    path = _find_run_json()
    if not path:
        print(
            "deepeval-pr-comment: no DeepEval test run JSON found "
            f"(looked in '{os.getenv('DEEPEVAL_CACHE_FOLDER', '.deepeval')}')",
            file=sys.stderr,
        )
        return 1

    run = load_test_run(path)
    body = format_summary(run, summary_only=summary_only)
    print(body)

    post_pr_comment(repo, pr_number, body, token)
    print("deepeval-pr-comment: posted comment to PR #" + str(pr_number))

    if fail_on_failure and (run.get("testFailed") or 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
