"""Build a Markdown PR comment from a DeepEval test run JSON.

The formatter is intentionally pure (no network, no DeepEval import) so it can
be unit tested against a fixture of the JSON that ``deepeval test run`` writes
to ``.deepeval/.latest_run_full.json``.
"""

from __future__ import annotations

import json
from typing import Any


def load_test_run(path: str) -> dict[str, Any]:
    """Load a DeepEval test-run JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_average(scores: list[float | None]) -> float:
    values = [s for s in scores if s is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _format_metrics_summary(run: dict[str, Any]) -> list[str]:
    metrics_scores = run.get("metricsScores") or []
    if not metrics_scores:
        return []

    lines = [
        "### Metrics",
        "",
        "| Metric | Avg Score | Pass | Fail | Error |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in metrics_scores:
        scores = metric.get("scores") or []
        avg = _safe_average(scores)
        lines.append(
            f"| {metric.get('metric')} "
            f"| {avg:.2f} "
            f"| {metric.get('passes', 0)} "
            f"| {metric.get('fails', 0)} "
            f"| {metric.get('errors', 0)} |"
        )
    return lines


def _format_test_cases(run: dict[str, Any]) -> list[str]:
    test_cases = run.get("testCases") or []
    conversational = run.get("conversationalTestCases") or []
    if not test_cases and not conversational:
        return []

    lines = ["### Test Cases", ""]
    rows = []

    for case in test_cases:
        rows.append(_format_single_case(case))

    for case in conversational:
        rows.append(_format_single_case(case))

    # Keep the comment readable: show every case but cap very long lists.
    shown = rows[:50]
    lines.extend(shown)
    if len(rows) > len(shown):
        lines.append("")
        lines.append(f"_… and {len(rows) - len(shown)} more test case(s)._")
    return lines


def _format_single_case(case: dict[str, Any]) -> str:
    name = case.get("name") or "(unnamed)"
    success = case.get("success")
    status = "✅" if success is True else ("❌" if success is False else "⚠️")
    metrics = case.get("metricsData") or []
    if not metrics:
        return f"- {status} **{name}**"
    parts = ", ".join(
        f"{m.get('name')}={m.get('score')}"
        for m in metrics
        if m.get("score") is not None
    )
    return f"- {status} **{name}** — {parts}"


def format_summary(run: dict[str, Any], *, summary_only: bool = False) -> str:
    """Render a DeepEval test run as a Markdown PR comment."""
    test_passed = run.get("testPassed") or 0
    test_failed = run.get("testFailed") or 0
    total = test_passed + test_failed
    duration = run.get("runDuration") or 0.0
    cost = run.get("evaluationCost")

    lines: list[str] = ["## 🧪 DeepEval Test Report", ""]
    header = (
        f"**{total}** test case(s) · "
        f"✅ {test_passed} passed · "
        f"❌ {test_failed} failed · "
        f"⏱ {duration:.1f}s"
    )
    if cost is not None:
        header += f" · 💲 {cost:.4f}"
    lines.append(header)
    lines.append("")

    lines.extend(_format_metrics_summary(run))
    if not summary_only:
        lines.extend(_format_test_cases(run))

    return "\n".join(lines).rstrip() + "\n"
