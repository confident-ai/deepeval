import pytest
import os
from rich import print
from typing import Optional, Any
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.test_run import TestRun, test_run_manager
import sys


def pytest_sessionstart(session: pytest.Session):
    test_run = TestRun(
        testFile=session.config.getoption("file_or_dir")[0],
        testCases=[],
        metricScores=[],
        configurations={},
    )
    test_run_manager.set_test_run(test_run)
    test_run_manager.save_test_run()


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: Optional[pytest.Item]
) -> Optional[Any]:
    os.environ[PYTEST_RUN_TEST_NAME] = item.nodeid.split("::")[-1]
    return None  # continue with the default protocol


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus):
    print("Running teardown with pytest sessionfinish...")

    yield


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    for report in terminalreporter.getreports("skipped"):
        if report.skipped:
            reason = report.longreprtext.split("\n")[-1]
            print(f"Test {report.nodeid} was skipped. Reason: {reason}")
