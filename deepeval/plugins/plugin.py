import pytest
import os
from rich import print
from deepeval.api import Api, TestRun
from typing import Optional, Any
from deepeval.constants import PYTEST_RUN_ENV_VAR, PYTEST_RUN_TEST_NAME


def pytest_sessionstart(session: pytest.Session):
    global test_filename
    test_run = TestRun(
        testFile=session.config.getoption("file_or_dir")[0],
        testCases=[],
        metricScores=[],
        configurations={},
    )
    test_filename = test_run.save()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: Optional[pytest.Item]
) -> Optional[Any]:
    os.environ[PYTEST_RUN_TEST_NAME] = item.nodeid.split("::")[-1]
    return None  # continue with the default protocol


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request: pytest.FixtureRequest) -> None:
    if hasattr(fixturedef, "fixturenamea"):
        name = fixturedef.fixturename
        if name == "run_configuration":
            if fixturedef.cached_result:
                fixture_value, _, _ = fixturedef.cached_result
                print("Fixture value: ", fixture_value)
                test_run: TestRun = TestRun.load(test_filename)
                test_run.configurations = fixture_value
                test_filename = test_run.save()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus):
    print("Running teardown with pytest sessionfinish...")
    # Code before yield will run before the test teardown

    # yield control back to pytest for the actual teardown
    yield

    # Code after yield will run after the test teardown
    test_run: TestRun = TestRun.load(test_filename)
    if os.getenv(PYTEST_RUN_ENV_VAR) and os.path.exists(".deepeval"):
        api: Api = Api()
        result = api.post_test_run(test_run)

    if test_run is None:
        return
    # Calculate the average of each metric
    metrics_avg = {
        metric.metric: metric.score for metric in test_run.metric_scores
    }
    # Count the number of passes and failures
    # Get all the possible metrics first
    all_metrics = {metric.metric for metric in test_run.metric_scores}

    # Loop through to filter for each metric
    passes = {
        metric: len(
            [
                test_case_metric
                for test_case in test_run.test_cases
                for test_case_metric in test_case.metrics_metadata
                if test_case_metric.metric == metric and test_case.success
            ]
        )
        for metric in all_metrics
    }
    failures = {
        metric: len(
            [
                test_case_metric
                for test_case in test_run.test_cases
                for test_case_metric in test_case.metrics_metadata
                if test_case_metric.metric == metric
            ]
        )
        - passes[metric]
        for metric in all_metrics
    }
    # Create a table with rich
    from rich.table import Table

    table = Table(title="Test Results")
    table.add_column("Metric", justify="right")
    table.add_column("Average Score", justify="right")
    table.add_column("Passes", justify="right")
    table.add_column("Failures", justify="right")
    table.add_column("Success Rate", justify="right")
    total_passes = 0
    total_failures = 0
    for metric, avg in metrics_avg.items():
        pass_count = passes[metric]
        fail_count = failures[metric]
        total_passes += pass_count
        total_failures += fail_count
        success_rate = pass_count / (pass_count + fail_count) * 100
        table.add_row(
            metric,
            str(avg),
            f"[green]{str(pass_count)}[/green]",
            f"[red]{str(fail_count)}[/red]",
            f"{success_rate:.2f}%",
        )
    total_tests = total_passes + total_failures
    overall_success_rate = total_passes / total_tests * 100
    table.add_row(
        "Total",
        "-",
        f"[green]{str(total_passes)}[/green]",
        f"[red]{str(total_failures)}[/red]",
        f"{overall_success_rate:.2f}%",
    )
    print(table)

    if os.getenv(PYTEST_RUN_ENV_VAR) and os.path.exists(".deepeval"):
        link = f"https://app.confident-ai.com/project/{result.projectId}/unit-tests/{result.testRunId}"
        print(
            "✅ Tests finished! View results on " f"[link={link}]{link}[/link]"
        )
    os.remove(test_filename)
