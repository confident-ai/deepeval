import pytest
import os
import json
from rich import print
from typing import Optional, Any
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.test_run import global_test_run_manager
from deepeval.utils import get_is_running_deepeval


def pytest_sessionstart(session: pytest.Session):
    is_running_deepeval = get_is_running_deepeval()

    if is_running_deepeval:
        global_test_run_manager.save_to_disk = True
        global_test_run_manager.create_test_run(
            file_name=session.config.getoption("file_or_dir")[0]
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: Optional[pytest.Item]
) -> Optional[Any]:
    os.environ[PYTEST_RUN_TEST_NAME] = item.nodeid.split("::")[-1]
    return None


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus):
    print("Running teardown with pytest sessionfinish...")

    yield


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    for report in terminalreporter.getreports("skipped"):
        if report.skipped:
            reason = report.longreprtext.split("\n")[-1]
            print(f"Test {report.nodeid} was skipped. Reason: {reason}")
