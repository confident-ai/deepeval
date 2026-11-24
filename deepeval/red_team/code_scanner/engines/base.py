import json
import re
from typing import Optional, Protocol

from ..schema import CodeFindingsList

HARNESS_PROVIDERS = ("codex", "claude-code", "cursor")
PROVIDERS = ("deepeval",) + HARNESS_PROVIDERS

IMPORT_NAME = {
    "codex": "openai_codex",
    "claude-code": "claude_agent_sdk",
    "cursor": "cursor_sdk",
}
PIP_NAME = {
    "codex": "openai-codex",
    "claude-code": "claude-agent-sdk",
    "cursor": "cursor-sdk",
}
ENV_KEY = {
    "codex": "OPENAI_API_KEY",
    "claude-code": "ANTHROPIC_API_KEY",
    "cursor": "CURSOR_API_KEY",
}


OUTPUT_FORMAT_INSTRUCTION = """

Analyze ONLY the code provided above. Do not access the filesystem, run \
commands, or use any tools.

Respond with ONLY a single JSON object and nothing else (no markdown fences, \
no prose). It MUST match this schema exactly:

{
  "findings": [
    {
      "filePath": "<repository-relative path>",
      "lineStart": <integer>,
      "lineEnd": <integer, optional>,
      "vulnerability": "<one of deepteam's vulnerability names>",
      "vulnerabilityType": "<the specific subtype>",
      "severity": "low | medium | high | critical",
      "reason": "<why this code is vulnerable>",
      "recommendation": "<suggested fix, optional>",
      "codeSnippet": "<the offending code excerpt, optional>"
    }
  ]
}

If there are no vulnerabilities, respond with {"findings": []}.
"""


class ScanEngine(Protocol):
    """
    Turns a scan prompt into a CodeFindingsList.
    """

    def generate_findings(self, prompt: str) -> CodeFindingsList: ...

    async def a_generate_findings(self, prompt: str) -> CodeFindingsList: ...


def missing_sdk(provider: str) -> ImportError:
    pkg = PIP_NAME[provider]
    return ImportError(
        f"The '{provider}' provider requires the {pkg} SDK. Install it with "
        f"`poetry add {pkg}` (or `pip install {pkg}`), or omit --provider to "
        f"use deepteam's built-in scanner."
    )


def extract_findings(text: Optional[str]) -> CodeFindingsList:
    """
    Parse a harness's free-form text answer into CodeFindingsList.
    """
    if not text or not text.strip():
        return CodeFindingsList(findings=[])

    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()

    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            data = json.loads(raw[start : end + 1])
    if data is None:
        raise ValueError("Could not parse findings JSON from harness output.")

    if isinstance(data, list):
        data = {"findings": data}
    return CodeFindingsList.model_validate(data)
