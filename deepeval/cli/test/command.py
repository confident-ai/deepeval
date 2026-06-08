import os
import sys
import time
from enum import Enum
from typing import Optional

import pytest
import typer
from typing_extensions import Annotated

from deepeval.telemetry import capture_evaluation_run
from deepeval.test_run import (
    TEMP_FILE_PATH,
    global_test_run_manager,
    invoke_test_run_end_hook,
)
from deepeval.config.settings import get_settings
from deepeval.test_run.cache import TEMP_CACHE_FILE_NAME
from deepeval.test_run.comparison import ComparisonReason
from deepeval.test_run.test_run import TestRunResultDisplay
from deepeval.utils import (
    delete_file_if_exists,
    set_identifier,
    set_is_running_deepeval,
    set_should_ignore_errors,
    set_should_skip_on_missing_params,
    set_should_use_cache,
    set_verbose_mode,
    set_test_regression_mode,
)

SETTINGS = get_settings()
app = typer.Typer(name="test")


class TestRunMode(str, Enum):
    DEFAULT = "default"
    REGRESSION = "regression"


def check_if_valid_file(test_file_or_directory: str):
    if "::" in test_file_or_directory:
        test_file_or_directory = test_file_or_directory.split("::", 1)[0]
    if os.path.isfile(test_file_or_directory) or os.path.isdir(
        test_file_or_directory
    ):
        return
    raise ValueError("Provided path is neither a valid file nor a directory.")


# Allow extra args and ignore unknown options allow extra args to be passed to pytest
@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def run(
    ctx: typer.Context,
    test_file_or_directory: str,
    color: str = "yes",
    durations: int = 10,
    pdb: bool = False,
    exit_on_first_failure: Annotated[
        bool, typer.Option("--exit-on-first-failure", "-x/-X")
    ] = False,
    show_warnings: Annotated[
        bool, typer.Option("--show-warnings", "-w/-W")
    ] = False,
    identifier: Optional[str] = typer.Option(
        None,
        "--identifier",
        "-id",
        help="Identify this test run with pytest",
    ),
    num_processes: Optional[int] = typer.Option(
        None,
        "--num-processes",
        "-n",
        help="Number of processes to use with pytest",
    ),
    repeat: Optional[int] = typer.Option(
        None,
        "--repeat",
        "-r",
        help="Number of times to rerun a test case",
    ),
    use_cache: Optional[bool] = typer.Option(
        False,
        "--use-cache",
        "-c",
        help="Whether to use cached results or not",
    ),
    ignore_errors: Optional[bool] = typer.Option(
        False,
        "--ignore-errors",
        "-i",
        help="Whether to ignore errors or not",
    ),
    skip_on_missing_params: Optional[bool] = typer.Option(
        False,
        "--skip-on-missing-params",
        "-s",
        help="Whether to skip test cases with missing parameters",
    ),
    verbose: Optional[bool] = typer.Option(
        None,
        "--verbose",
        "-v",
        help="Whether to turn on verbose mode for evaluation or not",
    ),
    display: Optional[TestRunResultDisplay] = typer.Option(
        TestRunResultDisplay.ALL.value,
        "--display",
        "-d",
        help="Whether to display all test cases or just some in the end",
        case_sensitive=False,
    ),
    mark: Optional[str] = typer.Option(
        None,
        "--mark",
        "-m",
        help="List of marks to run the tests with.",
    ),
    mode: TestRunMode = typer.Option(
        TestRunMode.DEFAULT,
        "--mode",
        help="Test run mode. 'regression' compares this run against the latest official run on Confident AI and exits 1 on regression. Defaults to 'default'.",
        case_sensitive=False,
    ),
):
    """Run a test"""
    test_regression = mode == TestRunMode.REGRESSION

    if test_regression and not SETTINGS.CONFIDENT_API_KEY:
        print(
            "Error: --mode regression requires a CONFIDENT_API_KEY environment variable to be set."
        )
        sys.exit(1)

    delete_file_if_exists(TEMP_FILE_PATH)
    delete_file_if_exists(TEMP_CACHE_FILE_NAME)
    check_if_valid_file(test_file_or_directory)
    set_is_running_deepeval(True)

    should_use_cache = use_cache and repeat is None
    set_should_use_cache(should_use_cache)
    set_should_ignore_errors(ignore_errors)
    set_should_skip_on_missing_params(skip_on_missing_params)
    set_verbose_mode(verbose)
    set_identifier(identifier)
    set_test_regression_mode(test_regression)

    global_test_run_manager.reset()

    pytest_args = [test_file_or_directory]

    if exit_on_first_failure:
        pytest_args.insert(0, "-x")

    pytest_args.extend(
        [
            "--verbose" if verbose else "--quiet",
            f"--color={color}",
            f"--durations={durations}",
            "-s",
        ]
    )

    if pdb:
        pytest_args.append("--pdb")
    if not show_warnings:
        pytest_args.append("--disable-warnings")
    if num_processes is not None:
        pytest_args.extend(["-n", str(num_processes)])

    if repeat is not None:
        pytest_args.extend(["--count", str(repeat)])
        if repeat < 1:
            raise ValueError("The repeat argument must be at least 1.")

    if mark:
        pytest_args.extend(["-m", mark])
    if identifier:
        pytest_args.extend(["--identifier", identifier])

    # Add the deepeval plugin file to pytest arguments
    pytest_args.extend(["-p", "deepeval"])
    # Append the extra arguments collected by allow_extra_args=True
    # Pytest will raise its own error if the arguments are invalid (error:
    if ctx.args:
        pytest_args.extend(ctx.args)

    start_time = time.perf_counter()
    with capture_evaluation_run("deepeval test run"):
        pytest_retcode = pytest.main(pytest_args)
    end_time = time.perf_counter()
    run_duration = end_time - start_time
    upload_result = global_test_run_manager.wrap_up_test_run(run_duration, True, display)

    invoke_test_run_end_hook()

    if test_regression:
        if upload_result is None:
            print(
                "Warning: test run was not uploaded to Confident AI — skipping baseline comparison."
            )
            sys.exit(0)

        _, run_id = upload_result
        test_run = global_test_run_manager.get_test_run()
        result = test_run.compare_with_official(run_id)
        _print_comparison_table(result)
        sys.exit(0 if result.passed else 1)

    if pytest_retcode == 1:
        sys.exit(1)

    return pytest_retcode


def _print_comparison_table(result) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import print as rprint

    console = Console()

    if result.reason == ComparisonReason.NO_OFFICIAL_RUN:
        rprint("\n[bold yellow]⚠ WARNING:[/bold yellow] No official run found for this project. Mark a run as official on Confident AI to enable regression testing.\n")
        return
    if result.reason == ComparisonReason.INCOMPATIBLE_RUNS:
        rprint("\n[bold red]✗ INCOMPATIBLE RUNS:[/bold red] One or more test cases from the official run were not found in the new run, or their metrics differ. Ensure you are running the same test cases with the same metrics.\n")
        return

    status_line = "[bold green]✓ NO REGRESSION DETECTED[/bold green]" if result.passed else "[bold red]✗ EVAL REGRESSION DETECTED[/bold red]"
    rprint(f"\n{status_line}")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Test Case", style="cyan", no_wrap=True)
    table.add_column("Metric")
    table.add_column("Official", justify="right")
    table.add_column("New", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("", justify="center")

    for tc in result.test_cases:
        for i, m in enumerate(tc.metrics):
            status = "[green]✓[/green]" if m.passed else "[red]✗[/red]"
            table.add_row(
                tc.name if i == 0 else "",
                m.metric,
                f"{m.official_score:.4f}",
                f"{m.new_score:.4f}",
                f"{m.delta:+.4f}",
                status,
            )

    console.print(table)

    if result.regressed_test_cases:
        rprint(
            f"[bold red]{result.regressed_test_cases}/{result.total_test_cases} test case(s) regressed.[/bold red]\n"
        )
    else:
        rprint(
            f"[bold green]All {result.total_test_cases} test case(s) within threshold.[/bold green]\n"
        )
