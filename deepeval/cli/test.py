import time
import pytest
import typer
import os
import sys
from typing_extensions import Annotated
from typing import Optional

from deepeval.test_run import global_test_run_manager, TEMP_FILE_PATH
from deepeval.test_run.cache import TEMP_CACHE_FILE_NAME
from deepeval.test_run.test_run import TestRunResultDisplay
from deepeval.utils import (
    delete_file_if_exists,
    set_should_ignore_errors,
    set_should_skip_on_missing_params,
    set_should_use_cache,
    set_verbose_mode,
    set_identifier,
)
from deepeval.test_run import invoke_test_run_end_hook
from deepeval.telemetry import capture_evaluation_run
from deepeval.utils import set_is_running_deepeval

app = typer.Typer(name="test")


def check_if_valid_file(test_file_or_directory: str):
    if "::" in test_file_or_directory:
        test_file_or_directory, test_case = test_file_or_directory.split("::")
    if os.path.isfile(test_file_or_directory):
        if test_file_or_directory.endswith(".py"):
            if not os.path.basename(test_file_or_directory).startswith("test_"):
                raise ValueError(
                    "Test will not run. Please ensure the file starts with `test_` prefix."
                )
    elif os.path.isdir(test_file_or_directory):
        return
    else:
        raise ValueError(
            "Provided path is neither a valid file nor a directory."
        )


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
):
    """Run a test"""
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
    global_test_run_manager.wrap_up_test_run(run_duration, True, display)

    invoke_test_run_end_hook()

    if pytest_retcode == 1:
        sys.exit(1)

    return pytest_retcode
