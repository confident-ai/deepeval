import pytest
import os
from rich import print
from typing import Optional, Any
from deepeval.constants import PYTEST_RUN_TEST_NAME
from deepeval.test_run import global_test_run_manager
from deepeval.utils import get_is_running_deepeval


def pytest_addoption(parser):
    parser.addoption(
        "--identifier",
        action="store",
        default=None,
        help="Custom identifier for the test run",
    )


def pytest_sessionstart(session: pytest.Session):
    is_running_deepeval = get_is_running_deepeval()
    identifier = session.config.getoption("identifier", None)

    if is_running_deepeval:
        global_test_run_manager.save_to_disk = True
        global_test_run_manager.create_test_run(
            identifier=identifier,
            file_name=session.config.getoption("file_or_dir")[0],
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: Optional[pytest.Item]
) -> Optional[Any]:
    os.environ[PYTEST_RUN_TEST_NAME] = item.nodeid.split("::")[-1]
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Wrap each test in a deepeval evaluation scope so `@observe` spans get
    attached to the in-flight test run via `assert_test(golden=..., metrics=...)`.
    """
    if not get_is_running_deepeval():
        yield
        return

    from deepeval.tracing.tracing import Observer, trace_manager

    prev_evaluating = trace_manager.evaluating
    trace_manager.evaluating = True
    observer = Observer("custom", func_name="Test Wrapper")
    observer.__enter__()
    try:
        yield
    finally:
        try:
            observer.__exit__(None, None, None)
        finally:
            trace_manager.evaluating = prev_evaluating


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus):
    print("Running teardown with pytest sessionfinish...")

    yield


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    for report in terminalreporter.getreports("skipped"):
        if report.skipped:
            reason = report.longreprtext.split("\n")[-1]
            print(f"Test {report.nodeid} was skipped. Reason: {reason}")
