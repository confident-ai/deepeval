"""`deepeval red-team`: red team from a YAML config, or scan code with --scan.

Only option parsing lives here. The red teaming modules are imported when the
command runs, so other CLI commands don't pay for them at startup.
"""

from typing import Optional

import typer

RED_TEAM_HELP = (
    "Red team an AI app from a YAML config, or scan source code for "
    "AI-security vulnerabilities with --scan PATH."
)


def red_team_command(
    config_file: Optional[str] = typer.Argument(
        None,
        help="YAML config describing the target, vulnerabilities and attacks.",
    ),
    scan: Optional[str] = typer.Option(
        None,
        "--scan",
        help="Scan a file or directory for AI-security vulnerabilities instead.",
    ),
    # red teaming
    max_concurrent: Optional[int] = typer.Option(
        None,
        "-c",
        "--max-concurrent",
        help="Maximum concurrent operations (overrides config).",
    ),
    attacks_per_vuln: Optional[int] = typer.Option(
        None,
        "-a",
        "--attacks-per-vuln",
        help="Number of attacks per vulnerability type (overrides config).",
    ),
    output_folder: Optional[str] = typer.Option(
        None,
        "-o",
        "--output-folder",
        help="Folder to save the risk assessment to (overrides config).",
    ),
    # code scanning
    diff: Optional[str] = typer.Option(
        None,
        "--diff",
        help="With --scan: only files changed between two git refs, e.g. 'main..HEAD'.",
    ),
    scan_config: Optional[str] = typer.Option(
        None,
        "--scan-config",
        help="With --scan: config YAML path (otherwise auto-discovered at the scan root).",
    ),
    output_format: str = typer.Option(
        "markdown",
        "-f",
        "--format",
        help="With --scan: markdown | sarif | json.",
    ),
    min_severity: Optional[str] = typer.Option(
        None,
        "--min-severity",
        help="With --scan: only report findings at or above low | medium | high | critical.",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "-p",
        "--provider",
        help=(
            "With --scan: codex | claude-code | cursor. Defaults from the API "
            "key that is set (OPENAI_API_KEY=codex, ANTHROPIC_API_KEY=claude-code, "
            "CURSOR_API_KEY=cursor), else the built-in scanner (OpenAI API)."
        ),
    ),
    model: Optional[str] = typer.Option(
        None,
        "-m",
        "--model",
        help="With --scan: model for the chosen provider. Overrides the config.",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        help="With --scan: write the report to a file instead of stdout.",
    ),
    fail_on_findings: bool = typer.Option(
        True,
        "--fail-on-findings/--no-fail-on-findings",
        help="With --scan: exit with code 1 if any findings remain (useful in CI).",
    ),
    comment: bool = typer.Option(
        False,
        "--comment",
        help="With --scan: post findings to Confident AI to comment on the PR (GitHub Actions only).",
    ),
    pr: Optional[int] = typer.Option(
        None,
        "--pr",
        help="With --scan: pull request number to comment on. Defaults to the PR from the GitHub Actions event.",
    ),
) -> None:
    if scan is not None and config_file is not None:
        raise typer.BadParameter(
            "Pass either a config file or --scan PATH, not both."
        )

    if scan is not None:
        from deepeval.cli.red_team.scan import scan as run_scan

        run_scan(
            path=scan,
            diff=diff,
            config_file=scan_config,
            output_format=output_format,
            min_severity=min_severity,
            provider=provider,
            model=model,
            output=output,
            fail_on_findings=fail_on_findings,
            comment=comment,
            pr=pr,
        )
        return

    if config_file is None:
        raise typer.BadParameter(
            "Pass a red teaming config file, or --scan PATH to scan code."
        )

    from deepeval.cli.red_team.run import run

    run(
        config_file,
        max_concurrent=max_concurrent,
        attacks_per_vuln=attacks_per_vuln,
        output_folder=output_folder,
    )
