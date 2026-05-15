import os
import sys
import glob
import time
from typing import List, Optional

import pytest
import typer
from typing_extensions import Annotated

from deepeval.telemetry import capture_evaluation_run
from deepeval.test_run import (
    TEMP_FILE_PATH,
    global_test_run_manager,
    invoke_test_run_end_hook,
)
from deepeval.test_run.cache import TEMP_CACHE_FILE_NAME
from deepeval.cli.utils import _post_github_pr_comment
from deepeval.test_run.test_run import TestRunResultDisplay
from deepeval.evaluate.console_report import EvaluationConsoleReport
from deepeval.utils import (
    delete_file_if_exists,
    set_identifier,
    set_is_running_deepeval,
    set_should_ignore_errors,
    set_should_skip_on_missing_params,
    set_should_use_cache,
    set_verbose_mode,
)

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
    pass_rate: Optional[float] = typer.Option(
        None, "--pass-rate", help="Minimum pass rate required (e.g., 0.8 for 80%)."
    ),
    required_metrics: Optional[List[str]] = typer.Option(
        None, "--required-metrics", help="List of metric names that MUST pass. Can be used multiple times."
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

    test_results = global_test_run_manager.get_test_run().test_results
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    
    if is_ci and test_results:
        print("CI environment detected. Generating PR comment...")
        
        passed = True
        actual_pass_rate = len([r for r in test_results if r.success]) / len(test_results)
        
        if pass_rate is not None and actual_pass_rate < pass_rate:
            passed = False
            
        if required_metrics:
            for test_result in test_results:
                for metric_data in test_result.metrics_data:
                    if metric_data.name in required_metrics and not metric_data.success:
                        passed = False
                        break

        report = EvaluationConsoleReport(test_results)
        output_dir = ".deepeval_ci_reports"
        report.export_to_cicd_markdown(output_dir=output_dir, evaluation_name="pr_evaluation")
        
        list_of_files = glob.glob(f"{output_dir}/*.md")
        latest_report_path = max(list_of_files, key=os.path.getctime)
        with open(latest_report_path, "r", encoding="utf-8") as f:
            markdown_summary = f.read()

        confident_link = global_test_run_manager.get_latest_test_run_link()
        if confident_link:
            markdown_summary += f"\n### 🔍 Deep Dive\n[**View the full results on Confident AI**]({confident_link})"
        else:
            markdown_summary += f"\nSet CONFIDENT_API_KEY to view these results on the Confident AI platform"

        if not passed:
            markdown_summary = markdown_summary.replace("**Status:** ✅ Passed", "**Status:** ❌ Failed (Thresholds not met)")
            pytest_retcode = 1

        # 4. Post Comment
        _post_github_pr_comment(markdown_summary)

    if pytest_retcode == 1:
        sys.exit(1)

    return pytest_retcode
