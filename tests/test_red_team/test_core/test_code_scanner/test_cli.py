import json

import pytest
from typer.testing import CliRunner

import deepteam.cli.main as climain
import deepteam.code_scanner.code_scanner as cs_module
from deepteam.code_scanner import CodeFinding, CodeFindingsList


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def scan_dir(tmp_path):
    (tmp_path / "agent.py").write_text("x = eval(resp)\n")
    return str(tmp_path)


@pytest.fixture(autouse=True)
def mock_llm(monkeypatch):
    # No real model, no real LLM call.
    monkeypatch.setattr(climain, "load_model", lambda spec: object())
    monkeypatch.setattr(cs_module, "initialize_model", lambda m: (m, False))
    monkeypatch.setattr(
        cs_module,
        "generate",
        lambda *a, **k: CodeFindingsList(
            findings=[
                CodeFinding(
                    filePath="agent.py",
                    lineStart=2,
                    vulnerability="Unexpected Code Execution",
                    vulnerabilityType="eval_usage",
                    severity="high",
                    reason="eval on input",
                    recommendation="use ast.literal_eval",
                )
            ]
        ),
    )


class TestScanCli:
    def test_json_and_exit_code(self, runner, scan_dir):
        r = runner.invoke(climain.app, ["scan", scan_dir, "--format", "json"])
        assert r.exit_code == 1  # findings present -> CI-failing
        assert json.loads(r.stdout)[0]["vulnerabilityType"] == "eval_usage"

    def test_markdown_default(self, runner, scan_dir):
        r = runner.invoke(climain.app, ["scan", scan_dir])
        assert "DeepTeam Code" in r.stdout

    def test_min_severity_filters(self, runner, scan_dir):
        r = runner.invoke(
            climain.app,
            [
                "scan",
                scan_dir,
                "--min-severity",
                "critical",
                "--format",
                "json",
            ],
        )
        assert r.exit_code == 0
        assert json.loads(r.stdout) == []

    def test_no_fail_flag(self, runner, scan_dir):
        r = runner.invoke(
            climain.app, ["scan", scan_dir, "--no-fail-on-findings"]
        )
        assert r.exit_code == 0

    def test_output_file(self, runner, scan_dir, tmp_path):
        out = str(tmp_path / "out.sarif")
        runner.invoke(
            climain.app, ["scan", scan_dir, "--format", "sarif", "-o", out]
        )
        with open(out) as f:
            assert json.load(f)["version"] == "2.1.0"

    def test_bad_format(self, runner, scan_dir):
        r = runner.invoke(climain.app, ["scan", scan_dir, "--format", "xml"])
        assert r.exit_code != 0

    def test_comment_flag_posts_findings(self, runner, scan_dir, monkeypatch):
        import deepteam.code_scanner as pkg

        calls = []
        monkeypatch.setattr(
            pkg, "post_pr_comments", lambda findings: calls.append(findings)
        )
        r = runner.invoke(
            climain.app,
            ["scan", scan_dir, "--comment", "--no-fail-on-findings"],
        )
        assert r.exit_code == 0
        assert len(calls) == 1 and len(calls[0]) == 1
