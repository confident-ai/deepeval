from typing import List, Optional

from deepteam.code_scanner.constants import (
    DEFAULT_CODE_SCAN_VULNERABILITIES,
)
from deepteam.code_scanner.taxonomy import (
    VulnerabilityRef,
    allowed_types,
    to_name,
)


class CodeScanTemplate:
    """
    The rubric for static AI-security code scanning.
    """

    @staticmethod
    def _render_catalog(vulnerabilities: List[VulnerabilityRef]) -> str:
        """
        Render `- Name: type1, type2` lines, pulling the allowed subtypes
        straight from the taxonomy so the judge can only use known labels.
        Entries may be vulnerability classes or name strings.
        """
        lines = []
        for vulnerability in vulnerabilities:
            name = to_name(vulnerability)
            types = allowed_types(name)
            if not types:
                continue
            lines.append(f"        - {name}: {', '.join(types)}")
        return "\n".join(lines)

    @staticmethod
    def generate_code_batch_evaluation(
        batch_data: str,
        vulnerabilities: Optional[List[VulnerabilityRef]] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """Build the evaluation prompt for one batch of code chunks.

        Args:
            batch_data: JSON array of code chunks. Each has `file_path`, `content`, optional `language`, and `start_line` (the absolute line number `content`'s first line maps to).
            vulnerabilities: Vulnerability classes or name strings to scan for. Defaults to `DEFAULT_CODE_SCAN_VULNERABILITIES`. Pass a subset to scope.
            instruction: Optional free-text project instruction (e.g. "treat any PII exposure as high severity").
        """
        vulnerabilities = vulnerabilities or DEFAULT_CODE_SCAN_VULNERABILITIES
        catalog = CodeScanTemplate._render_catalog(vulnerabilities)

        instruction_block = ""
        if instruction and instruction.strip():
            instruction_block = (
                "\n        ADDITIONAL PROJECT INSTRUCTION (apply on top of the "
                f"above):\n        {instruction.strip()}\n"
            )

        return f"""
        You are an expert application security reviewer specializing in AI / LLM
        applications. Review the following batch of source-code chunks and report
        ONLY vulnerabilities that match one of the categories below. This is
        STATIC analysis, reason about how the code could be exploited, not
        about any particular runtime input.

        VULNERABILITIES TO LOOK FOR (category: allowed types): {catalog}

        WHAT TO FOCUS ON (these are AI-app risks, not generic code smells):
        - Untrusted model output flowing into a dangerous sink: eval/exec, shell
          commands, SQL, file paths, or outbound HTTP built from an LLM's output.
        - Over-permissioned agents/tools: capabilities, permissions, or autonomy
          broader than the task needs.
        - Unsanitised external content (RAG documents, tool outputs, retrieved
          data) injected into prompts.
        - Hardcoded secrets / provider API keys / credentials in source -> report
          as "Prompt Leakage" / "secrets_and_credentials".
        - Exposed debug/admin endpoints, or missing authorization checks around
          actions the agent can take.

        CRITICAL INSTRUCTIONS:
        1. Flag ONLY an unambiguous, primary match to one of the categories AND
           one of its exact listed types. When in doubt, do NOT flag. Do not
           report generic style issues, performance, or non-AI code quality.
        2. Line numbers are ABSOLUTE. Each chunk carries `file_path` and a
           `start_line`; line K of a chunk's `content` is absolute line
           (start_line + K - 1). Report `filePath`, `lineStart`, and `lineEnd`
           against the real file.
        3. Severity reflects exploitability and impact: "low", "medium",
           "high", or "critical".
        4. `vulnerabilityType` MUST be one of the exact types listed for the
           chosen category above.
        5. Give a precise `reason`, an actionable `recommendation`, and the
           offending `codeSnippet`.{instruction_block}
        **
        EXAMPLE JSON OUTPUT

        Example with a finding:
        {{
            "findings": [
                {{
                    "filePath": "app/agent.py",
                    "lineStart": 42,
                    "lineEnd": 42,
                    "vulnerability": "Unexpected Code Execution",
                    "vulnerabilityType": "eval_usage",
                    "severity": "critical",
                    "reason": "eval() is called on text returned by the LLM, so a prompt-injected response can execute arbitrary Python.",
                    "recommendation": "Never eval LLM output; parse with ast.literal_eval or a strict schema.",
                    "codeSnippet": "result = eval(llm_response)"
                }}
            ]
        }}

        Example with no findings:
        {{
            "findings": []
        }}
        **

        Batch Data (JSON array of code chunks):
        {batch_data}

        Before returning, drop any finding that is not a direct, unambiguous
        match to one of the listed categories and one of its exact types.

        Return ONLY a JSON object with a 'findings' key containing a list of
        finding objects.

        JSON:
        """
