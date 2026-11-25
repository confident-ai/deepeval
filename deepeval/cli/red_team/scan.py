import asyncio
import os
from typing import Optional

import typer

from deepeval.cli.red_team.model_callback import load_model


def scan(
    path: str = ".",
    diff: Optional[str] = None,
    config_file: Optional[str] = None,
    output_format: str = "markdown",
    min_severity: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    output: Optional[str] = None,
    fail_on_findings: bool = True,
    comment: bool = False,
    pr: Optional[int] = None,
) -> None:
    """Scan source code for AI-security vulnerabilities."""
    from deepeval.red_team.code_scanner import (
        CodeScanner,
        build_engine,
        collect_changed_files,
        collect_files,
        filter_by_severity,
        load_config,
        post_pr_comments,
        resolve_provider,
        to_json,
        to_markdown,
        to_sarif,
    )

    formatters = {
        "markdown": to_markdown,
        "sarif": to_sarif,
        "json": to_json,
    }
    if output_format not in formatters:
        raise typer.BadParameter(
            f"format must be one of {', '.join(formatters)}"
        )

    scan_root = path if os.path.isdir(path) else os.path.dirname(path) or "."
    cfg = load_config(path=config_file, directory=scan_root)

    resolved_provider = resolve_provider(provider or cfg.provider)
    model_name = model or cfg.model
    engine = build_engine(resolved_provider, model_name)
    # deepeval's judge needs a deepeval model; harness engines manage their own.
    eval_model = load_model(model_name) if engine is None else None
    typer.echo(
        f"Scanning with provider '{resolved_provider}'"
        + (f" (model: {model_name})" if model_name else ""),
        err=True,
    )

    if diff:
        base, _, head = diff.partition("..")
        chunks = collect_changed_files(
            path,
            base=base,
            head=head or None,
            include=cfg.include,
            exclude=cfg.exclude,
        )
    else:
        chunks = collect_files(path, include=cfg.include, exclude=cfg.exclude)

    typer.echo(f"Scanning {len(chunks)} code chunk(s)...", err=True)

    scanner = CodeScanner(
        model=eval_model,
        engine=engine,
        vulnerabilities=cfg.vulnerabilities,
        instruction=cfg.instruction,
        # Harness engines spawn an agent per batch, so keep concurrency modest.
        max_concurrent=4 if engine is not None else 10,
    )
    # Harness engines benefit from concurrent batches; run them async.
    if engine is not None:
        findings = asyncio.run(scanner.a_scan(chunks))
    else:
        findings = scanner.scan(chunks)
    findings = filter_by_severity(findings, min_severity or cfg.min_severity)

    rendered = formatters[output_format](findings)
    if output:
        with open(output, "w") as f:
            f.write(rendered)
        typer.echo(f"Wrote {len(findings)} finding(s) to {output}", err=True)
    else:
        typer.echo(rendered)

    if comment:
        post_pr_comments(findings, pr_number=pr)

    if fail_on_findings and findings:
        raise typer.Exit(code=1)
