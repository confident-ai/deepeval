import time
import pytest
import typer
import os
import json
from typing_extensions import Annotated
from typing import Optional

from deepeval.test_run import test_run_manager, TEMP_FILE_NAME
from deepeval.test_run.cache import TEMP_CACHE_FILE_NAME
from deepeval.utils import (
    delete_file_if_exists,
    get_deployment_configs,
    set_should_ignore_errors,
    set_should_use_cache,
)
from deepeval.test_run import invoke_test_run_end_hook
from deepeval.telemetry import capture_evaluation_count
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


@app.command()
def run(
    test_file_or_directory: str,
    verbose: bool = True,
    color: str = "yes",
    durations: int = 10,
    pdb: bool = False,
    exit_on_first_failure: Annotated[
        bool, typer.Option("--exit-on-first-failure", "-x/-X")
    ] = False,
    show_warnings: Annotated[
        bool, typer.Option("--show-warnings", "-w/-W")
    ] = False,
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
    mark: Optional[str] = typer.Option(
        None,
        "--mark",
        "-m",
        help="List of marks to run the tests with.",
    ),
):
    """Run a test"""
    delete_file_if_exists(TEMP_FILE_NAME)
    delete_file_if_exists(TEMP_CACHE_FILE_NAME)
    check_if_valid_file(test_file_or_directory)
    set_is_running_deepeval(True)

    should_use_cache = use_cache and repeat is None
    set_should_use_cache(should_use_cache)
    set_should_ignore_errors(ignore_errors)

    test_run_manager.reset()

    pytest_args = [test_file_or_directory]

    if exit_on_first_failure:
        pytest_args.insert(0, "-x")

    deployment_configs = get_deployment_configs()
    if deployment_configs is not None:
        deployment_configs_json = json.dumps(deployment_configs)
        pytest_args.extend(["--deployment", deployment_configs_json])

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

    # Add the deepeval plugin file to pytest arguments
    pytest_args.extend(["-p", "plugins"])

    start_time = time.perf_counter()
    retcode = pytest.main(pytest_args)
    capture_evaluation_count()
    end_time = time.perf_counter()
    run_duration = end_time - start_time
    test_run_manager.wrap_up_test_run(run_duration)

    invoke_test_run_end_hook()

    return retcode
