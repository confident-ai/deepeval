"""Tests for the DeepEval PR-comment formatter and GitHub helpers."""

from __future__ import annotations

import json
import os

from deepeval_comment.formatter import format_summary, load_test_run
from deepeval_comment.github_client import extract_pr_number

SAMPLE_RUN = {
    "testPassed": 2,
    "testFailed": 1,
    "runDuration": 12.34,
    "evaluationCost": 0.0123,
    "metricsScores": [
        {
            "metric": "Answer Relevancy",
            "scores": [0.9, 0.85, 0.2],
            "passes": 2,
            "fails": 1,
            "errors": 0,
        },
        {
            "metric": "Faithfulness",
            "scores": [1.0, 0.95, 0.0],
            "passes": 2,
            "fails": 1,
            "errors": 0,
        },
    ],
    "testCases": [
        {
            "name": "weather question",
            "success": True,
            "metricsData": [
                {"name": "Answer Relevancy", "score": 0.9, "success": True},
                {"name": "Faithfulness", "score": 1.0, "success": True},
            ],
        },
        {
            "name": "math question",
            "success": False,
            "metricsData": [
                {"name": "Answer Relevancy", "score": 0.2, "success": False},
                {"name": "Faithfulness", "score": 0.0, "success": False},
            ],
        },
    ],
}


def test_format_summary_contains_totals_and_metrics(tmp_path):
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(SAMPLE_RUN))
    run = load_test_run(str(run_path))

    body = format_summary(run)

    assert "3** test case(s)" in body
    assert "✅ 2 passed" in body
    assert "❌ 1 failed" in body
    assert "Answer Relevancy" in body
    assert "Faithfulness" in body
    assert "weather question" in body
    assert "math question" in body
    # averages: Answer Relevancy = (0.9+0.85+0.2)/3 = 0.65
    assert "0.65" in body


def test_format_summary_summary_only_omits_test_cases(tmp_path):
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(SAMPLE_RUN))
    run = load_test_run(str(run_path))

    body = format_summary(run, summary_only=True)

    assert "weather question" not in body
    assert "### Metrics" in body


def test_format_summary_handles_missing_sections(tmp_path):
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps({"testPassed": 0, "testFailed": 0}))
    run = load_test_run(str(run_path))

    body = format_summary(run)

    assert "0** test case(s)" in body
    assert "### Metrics" not in body
    assert "### Test Cases" not in body


def test_extract_pr_number():
    assert extract_pr_number("refs/pull/123/merge") == "123"
    assert extract_pr_number("refs/heads/main") is None
    assert extract_pr_number("") is None
