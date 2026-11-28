import json
from typing import Dict, List

from .schema import CodeFinding

SEVERITY_ORDER: Dict[str, int] = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "critical": 3,
}

_SEVERITY_EMOJI = {
    "critical": "\U0001f534",
    "high": "\U0001f7e0",
    "medium": "\U0001f7e1",
    "low": "⚪",
}

_SARIF_LEVEL = {
    "critical": "error",
    "high": "error",
    "medium": "warning",
    "low": "note",
}


def filter_by_severity(
    findings: List[CodeFinding], min_severity: str = "low"
) -> List[CodeFinding]:
    """Keep only findings at or above `min_severity`."""
    threshold = SEVERITY_ORDER.get(min_severity, 0)
    return [
        f for f in findings if SEVERITY_ORDER.get(f.severity, 0) >= threshold
    ]


def to_json(findings: List[CodeFinding]) -> str:
    return json.dumps(
        [f.model_dump(exclude_none=True) for f in findings], indent=2
    )


def to_markdown(findings: List[CodeFinding]) -> str:
    """Render findings as a PR-comment-friendly Markdown summary."""
    if not findings:
        return (
            "## DeepTeam Code Vulnerabiltiy Scan\n\nNo vulnerabilities found."
        )

    counts = {s: 0 for s in SEVERITY_ORDER}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    summary = " · ".join(
        f"{_SEVERITY_EMOJI[s]} {counts[s]} {s}"
        for s in ("critical", "high", "medium", "low")
    )
    lines = [
        "## DeepTeam Code Vulnerabiltiy Scan",
        "",
        f"**{len(findings)} finding(s)**: {summary}",
    ]

    by_file: Dict[str, List[CodeFinding]] = {}
    for f in findings:
        by_file.setdefault(f.filePath, []).append(f)

    for path in sorted(by_file):
        lines.append(f"\n### `{path}`")
        ordered = sorted(
            by_file[path],
            key=lambda x: (-SEVERITY_ORDER.get(x.severity, 0), x.lineStart),
        )
        for f in ordered:
            loc = f"L{f.lineStart}"
            if f.lineEnd and f.lineEnd != f.lineStart:
                loc += f"-{f.lineEnd}"
            lines.append(
                f"- {_SEVERITY_EMOJI[f.severity]} **{f.severity.capitalize()}**: "
                f"`{f.vulnerability} / {f.vulnerabilityType}` [{loc}]"
            )
            lines.append(f"  {f.reason}")
            if f.recommendation:
                lines.append(f"  **Fix**: {f.recommendation}")

    return "\n".join(lines)


def to_sarif(findings: List[CodeFinding]) -> str:
    """Render findings as SARIF 2.1.0 for GitHub's code-scanning UI."""
    rules: Dict[str, dict] = {}
    results: List[dict] = []

    for f in findings:
        rule_id = f"{f.vulnerability}/{f.vulnerabilityType}"
        if rule_id not in rules:
            rules[rule_id] = {
                "id": rule_id,
                "name": f.vulnerability,
                "shortDescription": {
                    "text": f"{f.vulnerability}: {f.vulnerabilityType}"
                },
            }
        region = {"startLine": f.lineStart}
        if f.lineEnd:
            region["endLine"] = f.lineEnd
        results.append(
            {
                "ruleId": rule_id,
                "level": _SARIF_LEVEL.get(f.severity, "warning"),
                "message": {"text": f.reason},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": f.filePath},
                            "region": region,
                        }
                    }
                ],
            }
        )

    sarif = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "deepteam",
                        "informationUri": "https://github.com/confident-ai/deepteam",
                        "rules": list(rules.values()),
                    }
                },
                "results": results,
            }
        ],
    }
    return json.dumps(sarif, indent=2)
