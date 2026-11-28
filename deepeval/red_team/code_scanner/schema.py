from pydantic import BaseModel, Field
from typing import List, Literal, Optional

Severity = Literal["low", "medium", "high", "critical"]


class CodeChunk(BaseModel):
    """
    A unit of source code handed to the scanner for evaluation.
    """

    file_path: str = Field(
        ...,
        description="Repository-relative path of the file this code came from.",
    )
    content: str = Field(
        ...,
        description="The source code to scan.",
    )
    language: Optional[str] = Field(
        default=None,
        description="Detected language, e.g. 'python' or 'typescript'.",
    )
    start_line: int = Field(
        default=1,
        description=(
            "Absolute line number of the first line in `content`. 1 for a "
            "whole file; the hunk's starting line for a diff."
        ),
    )


class CodeFinding(BaseModel):
    """
    A single vulnerability located in source code.
    """

    filePath: str = Field(
        ...,
        description="Repository-relative path of the file containing the issue.",
    )
    lineStart: int = Field(
        ...,
        description="1-indexed line where the vulnerable code begins.",
    )
    lineEnd: Optional[int] = Field(
        default=None,
        description="1-indexed line where the vulnerable code ends, if multi-line.",
    )
    vulnerability: str = Field(
        ...,
        description=(
            "The vulnerability category. MUST be one of deepteam's known "
            "vulnerability names (see VULNERABILITY_CLASSES_MAP), e.g. "
            "'Unexpected Code Execution', 'SSRF', 'Excessive Agency'."
        ),
    )
    vulnerabilityType: str = Field(
        ...,
        description=(
            "The specific subtype within the category, drawn from that "
            "vulnerability's allowed types, e.g. 'eval_usage' or "
            "'shell_command_execution'."
        ),
    )
    severity: Severity = Field(
        ...,
        description="Impact rating: 'low', 'medium', 'high', or 'critical'.",
    )
    reason: str = Field(
        ...,
        description=(
            "Why this code is vulnerable: the specific construct and how it "
            "could be exploited."
        ),
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Concrete suggested fix for the issue.",
    )
    codeSnippet: Optional[str] = Field(
        default=None,
        description="The exact offending code excerpt.",
    )


class CodeFindingsList(BaseModel):
    findings: List[CodeFinding]
