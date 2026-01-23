"""
Pytest configuration for LangGraph integration tests.

- Uploads traces directly to Confident AI Observatory (/v1/traces) after each test.
- Also creates a TestRun with test cases for the Test Runs UI.
- Each test case includes trace_uuid in additional_metadata for correlation.
"""

import os
import pytest
import datetime

# Module-level state for TestRun
_test_run_identifier = None


def pytest_configure(config):
    """Set environment variables needed for upload."""
    os.environ["CONFIDENT_OPEN_BROWSER"] = "0"


def pytest_sessionstart(session: pytest.Session):
    """Create a TestRun at the start of the pytest session."""
    from deepeval.confident.api import is_confident

    if not is_confident():
        return

    from deepeval.test_run import global_test_run_manager

    global _test_run_identifier

    # Create a unique identifier for this test run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _test_run_identifier = f"langgraph-integrations-{timestamp}"

    # Enable disk persistence and create the test run
    global_test_run_manager.save_to_disk = True
    global_test_run_manager.create_test_run(
        identifier=_test_run_identifier,
        file_name="tests/test_integrations/test_langgraph",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    """After each test call phase, upload trace and add test case to TestRun."""
    outcome = yield
    report = outcome.get_result()

    # Only process after the test call phase (not setup/teardown)
    if call.when != "call":
        return

    from deepeval.confident.api import is_confident

    if not is_confident():
        return

    # Import the shared storage from utils
    from tests.test_integrations.utils import get_stored_trace

    trace_dict = get_stored_trace(item.nodeid)
    if trace_dict is None:
        return

    # 1) Upload trace directly to /v1/traces (keep existing logic)
    trace_uuid = _upload_trace_to_observatory(trace_dict)

    # 2) Add test case to TestRun with trace_uuid correlation
    if trace_uuid:
        _add_test_case_to_run(item.nodeid, report.passed, trace_uuid)


def _upload_trace_to_observatory(trace_dict: dict) -> str:
    """Upload trace dict directly to Confident AI Observatory via /v1/traces.

    Returns the trace UUID on success, None on failure.
    """
    from deepeval.confident.api import Api, Endpoints, HttpMethods

    trace_uuid = trace_dict.get("uuid", "unknown")

    try:
        api = Api()
        api.send_request(
            method=HttpMethods.POST,
            endpoint=Endpoints.TRACES_ENDPOINT,
            body=trace_dict,
        )
        print(f"UPLOADED TRACE UUID: {trace_uuid}")
        return trace_uuid
    except Exception as e:
        print(f"[ERROR] Failed to upload trace {trace_uuid}: {e}")
        return None


def _add_test_case_to_run(nodeid: str, passed: bool, trace_uuid: str):
    """Add a test case to the current TestRun with trace UUID correlation."""
    from deepeval.test_run import global_test_run_manager
    from deepeval.test_run.api import LLMApiTestCase
    from deepeval.test_case import LLMTestCase

    if global_test_run_manager.test_run is None:
        return

    # Create minimal test case (avoid large payloads)
    test_case = LLMTestCase(
        input=f"LangGraph test: {nodeid}",
        actual_output="PASSED" if passed else "FAILED",
    )

    # Create API test case with trace_uuid in additional_metadata for correlation
    # DO NOT set trace field - it causes 500 errors
    api_test_case = LLMApiTestCase(
        name=nodeid,
        input=f"LangGraph test: {nodeid}",
        actual_output="PASSED" if passed else "FAILED",
        success=passed,
        metrics_data=None,
        trace=None,  # Must be None - embedding traces causes 500s
        order=len(global_test_run_manager.test_run.test_cases),
        additional_metadata={
            "trace_uuid": trace_uuid,
            "pytest_nodeid": nodeid,
        },
    )

    global_test_run_manager.update_test_run(api_test_case, test_case)


def pytest_sessionfinish(session: pytest.Session, exitstatus):
    """Upload the TestRun at the end of the session."""
    from deepeval.confident.api import is_confident
    from deepeval.test_run import global_test_run_manager

    if not is_confident():
        return

    test_run = global_test_run_manager.test_run
    if test_run is None:
        return

    if (
        len(test_run.test_cases) == 0
        and len(test_run.conversational_test_cases) == 0
    ):
        return

    # Set required fields for API
    test_run.test_passed = sum(1 for tc in test_run.test_cases if tc.success)
    test_run.test_failed = sum(
        1 for tc in test_run.test_cases if not tc.success
    )

    try:
        result = global_test_run_manager.post_test_run(test_run)
        if result:
            link, run_id = result
            print(f"\nTEST RUN LINK: {link}\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to upload test run: {e}\n")
