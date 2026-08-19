import pytest
import os
import uuid
from contextlib import AbstractContextManager
from rich import print
from typing import Optional, Any
from deepeval.constants import (
    PYTEST_RUN_TEST_NAME,
    PYTEST_TRACE_TEST_WRAPPER_SPAN_NAME,
)
from deepeval.telemetry import (
    TELEMETRY_RUN_ID_ENV_VAR,
    Entrypoint,
    capture_evaluation_run,
)
from deepeval.test_run import global_test_run_manager
from deepeval.utils import get_is_running_deepeval


def pytest_addoption(parser):
    parser.addoption(
        "--identifier",
        action="store",
        default=None,
        help="Custom identifier for the test run",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Set before xdist spawns its workers, which happens at session start, so
    # every worker process inherits the id through the environment and the
    # whole `-n` session reports as one run.
    os.environ.setdefault(TELEMETRY_RUN_ID_ENV_VAR, str(uuid.uuid4()))


def _telemetry_belongs_here(config: pytest.Config) -> bool:
    """Whether this process should account for the session.

    Under `-n` the controller runs no tests, so counting it would add an empty
    run alongside the workers that did the work.
    """
    is_xdist_worker = hasattr(config, "workerinput")
    if is_xdist_worker:
        return True
    return getattr(config.option, "dist", "no") in (None, "no")


_run_scope: Optional[AbstractContextManager] = None


def pytest_sessionstart(session: pytest.Session):
    is_running_deepeval = get_is_running_deepeval()
    identifier = session.config.getoption("identifier", None)

    if is_running_deepeval:
        global_test_run_manager.save_to_disk = True
        global_test_run_manager.create_test_run(
            identifier=identifier,
            file_name=session.config.getoption("file_or_dir")[0],
        )

    # The scope lives here rather than in `deepeval test run` because this is
    # the process the tests actually run in. `skip_if_empty` keeps unrelated
    # pytest suites silent: the plugin auto-loads wherever deepeval is
    # installed, and a session with no deepeval assertions is not an
    # evaluation.
    global _run_scope
    if _run_scope is None and _telemetry_belongs_here(session.config):
        _run_scope = capture_evaluation_run(
            Entrypoint.PYTEST,
            run_id=os.environ.get(TELEMETRY_RUN_ID_ENV_VAR),
            skip_if_empty=True,
        )
        _run_scope.__enter__()


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
    print("Running teardown with pytest sessionfinish...")

    yield

    global _run_scope
    if _run_scope is not None:
        scope, _run_scope = _run_scope, None
        # A failing assertion is a normal evaluation result, not a failed run,
        # so the exit status is deliberately not turned into an outcome.
        scope.__exit__(None, None, None)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    for report in terminalreporter.getreports("skipped"):
        if report.skipped:
            reason = report.longreprtext.split("\n")[-1]
            print(f"Test {report.nodeid} was skipped. Reason: {reason}")
