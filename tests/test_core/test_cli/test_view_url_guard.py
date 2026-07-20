"""Regression tests for the ``deepeval view`` cached-link safety guard.

`is_confident_browser_url` gates the cached test-run link that `deepeval view`
opens in the browser, so a tampered local `.deepeval` state file cannot redirect
the browser to an arbitrary URL.
"""

import pytest

from deepeval.cli.utils import is_confident_browser_url

PROD = "https://app.confident-ai.com"


@pytest.mark.parametrize(
    "url",
    [
        f"{PROD}/test-runs/abc123",
        f"{PROD}/test-runs/abc123?tab=metrics",
        f"{PROD}/test-runs/abc123/spans",
    ],
)
def test_accepts_hosted_test_run_links(url):
    assert is_confident_browser_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "https://evil.com/test-runs/abc",  # wrong origin
        "https://app.confident-ai.com.evil.com/test-runs/abc",  # lookalike host
        f"{PROD}/settings",  # not a /test-runs/ path
        f"{PROD}/test-runsX/abc",  # prefix must be a full path segment
        "https://user:pw@app.confident-ai.com/test-runs/abc",  # embedded userinfo
        f"{PROD}/test-runs/../admin",  # dot-segment escape
        f"{PROD}/test-runs/%2e%2e/admin",  # encoded dot-segment escape
        "http://app.confident-ai.com/test-runs/abc",  # wrong scheme
        "javascript:alert(1)//test-runs/",  # non-http scheme, no host
    ],
)
def test_rejects_unsafe_links(url):
    assert is_confident_browser_url(url) is False
