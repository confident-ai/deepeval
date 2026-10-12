from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional
import typer
from rich import print

from deepeval.evaluate.compare_runs import (
    compare_runs,
    find_latest_runs,
    ComparativeConsoleReport,
)

def compare_command(
    path_a: Optional[Path] = typer.Argument(
        None,
        help="Path to the first test_run_*.json file (base run), or a folder containing runs. If folder or omitted, finds the second-latest run."
    ),
    path_b: Optional[Path] = typer.Argument(
        None,
        help="Path to the second test_run_*.json file (new run). If omitted, finds the latest run."
    ),
    folder: Optional[str] = typer.Option(
        None,
        "-f",
        "--folder",
        help="Folder to scan for test_run_*.json files. Overrides DEEPEVAL_RESULTS_FOLDER."
    ),
    file_type: Optional[str] = typer.Option(
        None,
        "-t",
        "--file-type",
        help="Export comparison report to a file format: 'html', 'md', or 'json'."
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "-o",
        "--output-dir",
        help="Directory or file path to save the exported report file. Set to '-' to output md/json directly to stdout."
    ),
    status: str = typer.Option(
        "all",
        "--status",
        "-s",
        help="Filter detailed test case breakdowns in the console output: 'all', 'regressed', 'improved'."
    ),
    fail_on_regression: bool = typer.Option(
        False,
        "-r",
        "--fail-on-regression",
        help="Exit with a non-zero code if any metric degraded or test case regressed from PASS to FAIL."
    ),
) -> None:
    """Compare two local test run JSON files side-by-side."""

    # 1. Validate status parameter
    valid_statuses = ["all", "regressed", "improved"]
    if status.lower() not in valid_statuses:
        print(f"[red]Error: Invalid status filter '{status}'. Supported: 'all', 'regressed', 'improved'[/red]")
        raise typer.Exit(code=1)

    # 2. Resolve paths
    run_a_path: Optional[Path] = None
    run_b_path: Optional[Path] = None

    if path_a is not None and path_b is not None:
        if not path_a.is_file():
            print(f"[red]Error: {path_a} is not a valid file.[/red]")
            raise typer.Exit(code=1)
        if not path_b.is_file():
            print(f"[red]Error: {path_b} is not a valid file.[/red]")
            raise typer.Exit(code=1)
        run_a_path = path_a
        run_b_path = path_b
    else:
        scan_folder = None
        if folder:
            scan_folder = Path(folder)
        elif path_a and path_a.is_dir():
            scan_folder = path_a
        else:
            env_folder = os.getenv("DEEPEVAL_RESULTS_FOLDER")
            if env_folder:
                scan_folder = Path(env_folder)
            else:
                legacy = Path("experiments")
                if legacy.is_dir():
                    scan_folder = legacy

        if not scan_folder or not scan_folder.is_dir():
            print(
                "[red]Error: Could not resolve a valid folder containing test runs. "
                "Please specify --folder, set DEEPEVAL_RESULTS_FOLDER, or provide explicit file paths.[/red]"
            )
            raise typer.Exit(code=1)

        try:
            resolved_a, resolved_b = find_latest_runs(scan_folder)
            if path_a and path_a.is_file():
                run_a_path = path_a
                run_b_path = resolved_b
            else:
                run_a_path = resolved_a
                run_b_path = resolved_b
        except Exception as e:
            print(f"[red]Error resolving test runs in {scan_folder}: {e}[/red]")
            raise typer.Exit(code=1)

    # Skip verbose printing if streaming to stdout
    is_streaming = output_dir == "-" and file_type and file_type.lower() in ["md", "json"]
    
    if not is_streaming:
        print(f"⚖️ Comparing runs:\n  [dim]Base (A):[/dim] {run_a_path}\n  [dim]New  (B):[/dim] {run_b_path}\n")

    # 3. Run comparison
    try:
        comparison = compare_runs(run_a_path, run_b_path)
    except Exception as e:
        print(f"[red]Error comparing runs: {e}[/red]")
        raise typer.Exit(code=1)

    # 4. Render report to console (if not streaming)
    report = ComparativeConsoleReport(comparison)
    if not is_streaming:
        report.render_to_terminal(status_filter=status.lower())

    # 5. Handle exports
    if file_type:
        out_dir = output_dir or "."
        f_type = file_type.lower()
        if f_type == "html":
            if out_dir == "-":
                print("[red]Error: HTML export cannot be streamed to stdout.[/red]")
                raise typer.Exit(code=1)
            report.export_to_html(out_dir, evaluation_name=f"comparison_{comparison.run_b_name}")
        elif f_type == "md":
            report.export_to_markdown(out_dir, evaluation_name=f"comparison_{comparison.run_b_name}")
        elif f_type == "json":
            report.export_to_json(out_dir, evaluation_name=f"comparison_{comparison.run_b_name}")
        else:
            print(f"[red]Error: Invalid file-type '{file_type}'. Supported: 'html', 'md', 'json'[/red]")
            raise typer.Exit(code=1)

    # 6. Handle fail_on_regression exit code
    if fail_on_regression:
        has_regression = False
        for diff in comparison.case_diffs:
            if diff.change_status == "degraded":
                has_regression = True
                break
        
        if has_regression:
            if not is_streaming:
                print("[bold red]❌ Regressions detected! Exiting with non-zero code.[/bold red]")
            raise typer.Exit(code=1)
        else:
            if not is_streaming:
                print("[bold green]✅ No regressions detected.[/bold green]")
