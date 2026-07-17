import os
from typing import Any, Optional, cast

import pytest


PYTEST_RUN_TEST_NAME = "CONFIDENT_AI_RUN_TEST_NAME"
PYTEST_TRACE_TEST_WRAPPER_SPAN_NAME = (
    "__deepeval_internal_pytest_test_wrapper__"
)
_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "on", "enable", "enabled"})


def _clear_run_test_name() -> None:
    os.environ.pop(PYTEST_RUN_TEST_NAME, None)


def get_is_running_deepeval() -> bool:
    value = os.getenv("DEEPEVAL")
    if value is None:
        return False

    return str(value).strip().strip('"').strip("'").lower() in _TRUTHY


def pytest_addoption(parser):
    if not get_is_running_deepeval():
        return

    parser.addoption(
        "--identifier",
        action="store",
        default=None,
        help="Custom identifier for the test run",
    )


def pytest_configure(config: pytest.Config):
    if not get_is_running_deepeval():
        _clear_run_test_name()


def pytest_sessionstart(session: pytest.Session):
    if not get_is_running_deepeval():
        _clear_run_test_name()
        return

    from deepeval.test_run import global_test_run_manager

    identifier = cast(
        Optional[str], session.config.getoption("identifier", None)
    )
    file_or_dir = cast(list[str], session.config.getoption("file_or_dir"))
    global_test_run_manager.save_to_disk = True
    global_test_run_manager.create_test_run(
        identifier=identifier,
        file_name=file_or_dir[0],
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(
    item: pytest.Item, nextitem: Optional[pytest.Item]
) -> Optional[Any]:
    if not get_is_running_deepeval():
        _clear_run_test_name()
        return None

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
    from deepeval.tracing.types import EvalMode, EvalSession

    prev_session = trace_manager.eval_session
    trace_manager.eval_session = EvalSession(mode=EvalMode.EVALUATE)
    observer = Observer("custom", func_name=PYTEST_TRACE_TEST_WRAPPER_SPAN_NAME)
    observer.__enter__()
    try:
        yield
    finally:
        try:
            observer.__exit__(None, None, None)
        finally:
            trace_manager.eval_session = prev_session


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus):
    if get_is_running_deepeval():
        from rich import print

        print("Running teardown with pytest sessionfinish...")
    else:
        _clear_run_test_name()

    try:
        yield
    finally:
        _clear_run_test_name()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not get_is_running_deepeval():
        return

    from rich import print

    for report in terminalreporter.getreports("skipped"):
        if report.skipped:
            reason = report.longreprtext.split("\n")[-1]
            print(f"Test {report.nodeid} was skipped. Reason: {reason}")


if not get_is_running_deepeval():
    _clear_run_test_name()
