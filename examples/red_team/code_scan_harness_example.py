"""
Smoke-test the DeepTeam code scanner through an agentic harness, using that vendor's own API key.

Each harness SDK is an optional extra. Install the one you want and set its key:

    poetry add "deepteam[codex]"       && export OPENAI_API_KEY=...      # Codex
    poetry add "deepteam[claude-code]" && export ANTHROPIC_API_KEY=...   # Claude Code
    poetry add "deepteam[cursor]"      && export CURSOR_API_KEY=...      # Cursor CLI

Then run this script, naming the provider (and optionally a model):

    poetry run python examples/code_scan_harness_example.py codex
    poetry run python examples/code_scan_harness_example.py claude-code <model>
    poetry run python examples/code_scan_harness_example.py cursor
    poetry run python examples/code_scan_harness_example.py

The exact same selection works on the CLI:

    poetry run deepteam scan . --provider codex
    poetry run deepteam scan . --provider claude-code --model <model>
"""

import sys

from deepteam.code_scanner import (
    CodeChunk,
    CodeScanner,
    build_engine,
    resolve_provider,
)

VULNERABLE_CODE = """\
import subprocess


def handle(llm_output: str):
    eval(llm_output)
    subprocess.run(llm_output, shell=True)
    api_key = "sk-this-should-not-be-here-1234567890"
    return api_key
"""


def main() -> None:
    provider = resolve_provider(sys.argv[1] if len(sys.argv) > 1 else None)
    model = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Provider: {provider}  |  Model: {model or '(provider default)'}\n")

    engine = build_engine(provider, model)
    scanner = CodeScanner(engine=engine)

    chunks = [
        CodeChunk(
            file_path="agent.py",
            content=VULNERABLE_CODE,
            language="python",
        )
    ]
    findings = scanner.scan(chunks)

    print(f"{len(findings)} finding(s):")
    for finding in findings:
        loc = f"{finding.filePath}:{finding.lineStart}"
        print(
            f"  [{finding.severity.upper():8}] {loc}  "
            f"{finding.vulnerability} / {finding.vulnerabilityType}"
        )
        print(f"             {finding.reason}")
        if finding.recommendation:
            print(f"             fix: {finding.recommendation}")


if __name__ == "__main__":
    main()
