import json

from deepteam.code_scanner import (
    CodeFinding,
    filter_by_severity,
    to_json,
    to_markdown,
    to_sarif,
)


def _f(severity, ln=1, recommendation=None):
    return CodeFinding(
        filePath="a.py",
        lineStart=ln,
        vulnerability="SSRF",
        vulnerabilityType="port_scanning",
        severity=severity,
        reason="r",
        recommendation=recommendation,
    )


class TestReport:
    def test_filter_by_severity(self):
        fs = [_f("low"), _f("high"), _f("critical")]
        assert len(filter_by_severity(fs, "low")) == 3
        assert len(filter_by_severity(fs, "high")) == 2
        assert len(filter_by_severity(fs, "critical")) == 1

    def test_to_json_roundtrips(self):
        d = json.loads(to_json([_f("high")]))
        assert d[0]["vulnerability"] == "SSRF"
        assert d[0]["severity"] == "high"

    def test_markdown_empty(self):
        assert "No vulnerabilities found" in to_markdown([])

    def test_markdown_has_finding(self):
        md = to_markdown([_f("critical", recommendation="do x")])
        assert "SSRF / port_scanning" in md
        assert "**Critical**" in md  # severity badge is capitalized
        assert "do x" in md

    def test_sarif_valid_and_level_mapping(self):
        s = json.loads(to_sarif([_f("critical"), _f("low", ln=9)]))
        assert s["version"] == "2.1.0"
        results = s["runs"][0]["results"]
        assert results[0]["level"] == "error"  # critical -> error
        assert results[1]["level"] == "note"  # low -> note
        region = results[1]["locations"][0]["physicalLocation"]["region"]
        assert region["startLine"] == 9
