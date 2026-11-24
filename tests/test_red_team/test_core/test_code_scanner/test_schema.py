import pytest
from pydantic import ValidationError

from deepteam.code_scanner import CodeChunk, CodeFinding, CodeFindingsList


class TestSchema:
    def test_code_finding_minimal(self):
        f = CodeFinding(
            filePath="a.py",
            lineStart=1,
            vulnerability="SSRF",
            vulnerabilityType="port_scanning",
            severity="high",
            reason="x",
        )
        assert f.filePath == "a.py"
        assert f.severity == "high"
        # Optional fields default to None and drop from an exclude_none dump.
        assert f.recommendation is None
        assert "recommendation" not in f.model_dump(exclude_none=True)

    def test_code_finding_full_roundtrip(self):
        f = CodeFinding(
            filePath="a.py",
            lineStart=1,
            lineEnd=3,
            vulnerability="Unexpected Code Execution",
            vulnerabilityType="eval_usage",
            severity="critical",
            reason="x",
            recommendation="y",
            codeSnippet="eval(z)",
        )
        d = f.model_dump(exclude_none=True)
        assert d["lineEnd"] == 3
        assert d["codeSnippet"] == "eval(z)"

    def test_invalid_severity_rejected(self):
        with pytest.raises(ValidationError):
            CodeFinding(
                filePath="a.py",
                lineStart=1,
                vulnerability="SSRF",
                vulnerabilityType="port_scanning",
                severity="extreme",
                reason="x",
            )

    def test_findings_list(self):
        assert CodeFindingsList(findings=[]).findings == []

    def test_code_chunk_defaults(self):
        c = CodeChunk(file_path="a.py", content="x=1")
        assert c.start_line == 1
        assert c.language is None
