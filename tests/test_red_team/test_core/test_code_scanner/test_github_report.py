import json

import pytest

import deepteam.code_scanner.github.report as gr
from deepteam.code_scanner import CodeFinding


class _Resp:
    def __init__(self, data=None):
        self._data = data or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def _finding():
    return CodeFinding(
        filePath="a.py",
        lineStart=1,
        vulnerability="SSRF",
        vulnerabilityType="port_scanning",
        severity="high",
        reason="x",
    )


def _setup_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "req-tok")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://oidc/req")
    monkeypatch.setenv("GITHUB_REPOSITORY", "octo/repo")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 42}}))
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setattr(gr, "get_base_api_url", lambda: "https://api.test")


class TestGithubReport:
    def test_posts_payload_with_oidc(self, monkeypatch, tmp_path):
        _setup_env(monkeypatch, tmp_path)
        monkeypatch.setattr(
            gr.requests,
            "get",
            lambda url, headers=None, timeout=None: _Resp({"value": "jwt"}),
        )
        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

        monkeypatch.setattr(gr.requests, "post", fake_post)

        gr.post_pr_comments([_finding()])

        assert (
            captured["url"]
            == "https://api.test/v1/github-app/code-scan/comments"
        )
        body = captured["json"]
        assert body["app"] == "deepteam"
        assert body["repo"] == "octo/repo"
        assert body["pr"] == 42
        assert body["oidc"] == "jwt"
        assert body["findings"][0]["vulnerability"] == "SSRF"

    def test_pr_number_from_ref(self, monkeypatch):
        monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
        monkeypatch.setenv("GITHUB_REF", "refs/pull/77/merge")
        assert gr._pr_number() == 77

    def test_raises_without_oidc(self, monkeypatch, tmp_path):
        _setup_env(monkeypatch, tmp_path)
        monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", raising=False)
        with pytest.raises(RuntimeError):
            gr.post_pr_comments([_finding()])

    def test_raises_without_repo(self, monkeypatch, tmp_path):
        _setup_env(monkeypatch, tmp_path)
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        monkeypatch.setattr(
            gr.requests,
            "get",
            lambda url, headers=None, timeout=None: _Resp({"value": "jwt"}),
        )
        with pytest.raises(RuntimeError):
            gr.post_pr_comments([_finding()])
