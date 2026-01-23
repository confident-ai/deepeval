"""
Pytest configuration for LangGraph integration tests.

- Uploads traces directly to Confident AI Observatory (/v1/traces) after each test.
- Also creates a TestRun with test cases for the Test Runs UI.
- Each test case includes trace_uuid in additional_metadata for correlation.
- Test case fields (input, actual_output, tools_called) are derived from trace_dict.
"""

import os
import sys
import pytest
import datetime
from typing import Dict, Any, List, Optional

from deepeval.test_case import ToolCall

# Module-level state for TestRun
_test_run_identifier = None

# Max length for input/output strings to avoid large payloads
MAX_FIELD_LENGTH = 2000


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

    # 2) Add test case to TestRun with data extracted from trace_dict
    if trace_uuid:
        _add_test_case_to_run(
            item.nodeid, report.passed, trace_uuid, trace_dict
        )


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


def _truncate(s: str, max_len: int = MAX_FIELD_LENGTH) -> str:
    """Truncate string to max_len, adding ellipsis if truncated."""
    if s and len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _extract_input_from_trace(trace_dict: Dict[str, Any]) -> str:
    """Extract a readable input string from trace_dict.

    Prefers messages[0].content if present, otherwise stringifies trace_dict["input"].
    """
    trace_input = trace_dict.get("input")
    if trace_input is None:
        return ""

    # If input has messages array, extract first message content
    if isinstance(trace_input, dict) and "messages" in trace_input:
        messages = trace_input.get("messages", [])
        if messages and isinstance(messages[0], dict):
            content = messages[0].get("content", "")
            if content:
                return _truncate(str(content))

    # Fallback: stringify the input
    return _truncate(str(trace_input))


def _extract_output_from_trace(trace_dict: Dict[str, Any]) -> str:
    """Extract a readable output string from trace_dict.

    Prefers last message content if present, otherwise stringifies trace_dict["output"].
    """
    trace_output = trace_dict.get("output")
    if trace_output is None:
        return ""

    # If output has messages array, extract last message content
    if isinstance(trace_output, dict) and "messages" in trace_output:
        messages = trace_output.get("messages", [])
        if messages:
            # Find last AI message with content
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("type") == "ai":
                    content = msg.get("content", "")
                    if content:
                        return _truncate(str(content))
            # Fallback to last message regardless of type
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
                if content:
                    return _truncate(str(content))

    # Fallback: stringify the output
    return _truncate(str(trace_output))


def _extract_tools_called_from_trace(
    trace_dict: Dict[str, Any]
) -> Optional[List[ToolCall]]:
    """Extract tools_called from trace_dict.

    Uses toolsCalled if present, otherwise derives from toolSpans.
    Returns list of ToolCall objects or None.
    """
    from deepeval.test_case import ToolCall

    result = []

    # First try top-level toolsCalled
    tools_called = trace_dict.get("toolsCalled")
    if tools_called and isinstance(tools_called, list):
        for tc in tools_called:
            if isinstance(tc, dict):
                try:
                    result.append(
                        ToolCall(
                            name=tc.get("name", "unknown_tool"),
                            input_parameters=tc.get("inputParameters")
                            or tc.get("input_parameters"),
                            output=(
                                _truncate(str(tc.get("output")))
                                if tc.get("output")
                                else None
                            ),
                        )
                    )
                except Exception:
                    pass

    # If no toolsCalled, try toolSpans
    if not result:
        tool_spans = trace_dict.get("toolSpans", [])
        for span in tool_spans:
            if isinstance(span, dict):
                try:
                    tool_input = span.get("input")
                    tool_output = span.get("output")
                    result.append(
                        ToolCall(
                            name=span.get("name", "unknown_tool"),
                            input_parameters=(
                                tool_input
                                if isinstance(tool_input, dict)
                                else None
                            ),
                            output=(
                                _truncate(str(tool_output))
                                if tool_output
                                else None
                            ),
                        )
                    )
                except Exception:
                    pass

    return result if result else None


def _get_environment_info() -> Dict[str, str]:
    """Collect environment info for debugging."""
    info = {
        "python_version": sys.version.split()[0],
    }

    # Try to get langchain/langgraph versions
    try:
        import langchain_core

        info["langchain_core_version"] = getattr(
            langchain_core, "__version__", "unknown"
        )
    except ImportError:
        pass

    try:
        import langgraph

        info["langgraph_version"] = getattr(langgraph, "__version__", "unknown")
    except ImportError:
        pass

    try:
        import langchain_openai

        info["langchain_openai_version"] = getattr(
            langchain_openai, "__version__", "unknown"
        )
    except ImportError:
        pass

    return info


def _add_test_case_to_run(
    nodeid: str, passed: bool, trace_uuid: str, trace_dict: Dict[str, Any]
):
    """Add a test case to the current TestRun with data extracted from trace_dict.

    NOTE: We bypass global_test_run_manager.update_test_run() and directly call
    test_run.add_test_case() because update_test_run has a guard that silently
    returns when metrics_data is empty AND trace is None:

        if (
            api_test_case.metrics_data is not None
            and len(api_test_case.metrics_data) == 0
            and api_test_case.trace is None
        ):
            return  # <-- never adds the test case!

    For integration tests without metrics evaluation, we must bypass this guard.
    We set metrics_data=None to signal "no metrics evaluated" (vs empty list
    meaning "metrics evaluated but found none"), and directly add the test case.
    """
    from deepeval.test_run import global_test_run_manager
    from deepeval.test_run.api import LLMApiTestCase

    test_run = global_test_run_manager.test_run
    if test_run is None:
        return

    # Parse nodeid for metadata
    # Format: tests/path/to/test.py::TestClass::test_method
    parts = nodeid.split("::")
    test_file = parts[0] if parts else nodeid
    test_name = parts[-1] if parts else nodeid

    # Extract fields from trace_dict
    input_str = _extract_input_from_trace(trace_dict)
    output_str = _extract_output_from_trace(trace_dict)
    tools_called = _extract_tools_called_from_trace(trace_dict)

    # Build additional_metadata with correlation and environment info
    additional_metadata = {
        "trace_uuid": trace_uuid,
        "pytest_nodeid": nodeid,
        "test_file": test_file,
        "test_name": test_name,
        "trace_name": trace_dict.get("name"),
        **_get_environment_info(),
    }

    # Determine order (index) for this test case
    order = len(test_run.test_cases)

    # Build LLMApiTestCase directly with camelCase field aliases.
    # We set metricsData=None (not []) to avoid the guard in update_test_run,
    # and trace=None to avoid server 500 errors when embedding traces.
    api_test_case = LLMApiTestCase(
        name=nodeid,
        input=input_str or f"LangGraph test: {test_name}",
        actualOutput=output_str or ("PASSED" if passed else "FAILED"),
        toolsCalled=tools_called,
        additionalMetadata=additional_metadata,
        success=passed,
        metricsData=None,  # None = "no metrics evaluated" (bypasses guard)
        trace=None,  # Must be None - embedding traces causes 500s
        order=order,
        runDuration=0,
        evaluationCost=None,
    )

    # Debug: print serialized payload keys for verification
    try:
        payload = api_test_case.model_dump(by_alias=True, exclude_none=True)
        print(f"[DEBUG] Test case payload keys: {list(payload.keys())}")
        print(f"[DEBUG]   actualOutput present: {'actualOutput' in payload}")
        print(f"[DEBUG]   toolsCalled present: {'toolsCalled' in payload}")
        print(
            f"[DEBUG]   additionalMetadata present: {'additionalMetadata' in payload}"
        )
        if "actualOutput" in payload:
            print(
                f"[DEBUG]   actualOutput value (truncated): {str(payload['actualOutput'])[:100]}..."
            )
        if "toolsCalled" in payload:
            print(f"[DEBUG]   toolsCalled count: {len(payload['toolsCalled'])}")

    except Exception as e:
        print(f"[DEBUG] Error dumping payload: {e}")

    # Directly add to test_run.test_cases, bypassing update_test_run guard
    test_run.add_test_case(api_test_case)
    print(
        f"[DEBUG] after add_test_case, test_cases: {len(test_run.test_cases)}"
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus):
    """Upload the TestRun at the end of the session."""
    from deepeval.confident.api import is_confident
    from deepeval.test_run import global_test_run_manager

    if not is_confident():
        print("[DEBUG] sessionfinish is_confident:", is_confident())
        return

    test_run = global_test_run_manager.test_run
    if test_run is None:
        print("[DEBUG] sessionfinish right after getting test_run")
        print("[DEBUG] sessionfinish is_confident:", is_confident())
        print("[DEBUG] sessionfinish test_run is None:", test_run is None)
        print(
            "[DEBUG] sessionfinish test_cases:",
            len(test_run.test_cases) if test_run else None,
        )
        print(
            "[DEBUG] sessionfinish conversational:",
            len(test_run.conversational_test_cases) if test_run else None,
        )
        print(
            "[DEBUG] sessionfinish identifier:",
            getattr(test_run, "identifier", None),
        )

        return

    if (
        len(test_run.test_cases) == 0
        and len(test_run.conversational_test_cases) == 0
    ):
        print("[DEBUG] sessionfinish checking test_cases length is not 0")
        print("[DEBUG] sessionfinish is_confident:", is_confident())
        print("[DEBUG] sessionfinish test_run is None:", test_run is None)
        print(
            "[DEBUG] sessionfinish test_cases:",
            len(test_run.test_cases) if test_run else None,
        )
        print(
            "[DEBUG] sessionfinish conversational:",
            len(test_run.conversational_test_cases) if test_run else None,
        )
        print(
            "[DEBUG] sessionfinish identifier:",
            getattr(test_run, "identifier", None),
        )
        return

    # Set required fields for API
    test_run.test_passed = sum(1 for tc in test_run.test_cases if tc.success)
    test_run.test_failed = sum(
        1 for tc in test_run.test_cases if not tc.success
    )

    try:
        result = global_test_run_manager.post_test_run(test_run)
        print("[DEBUG] post_test_run returned:", result)
        if result:
            link, run_id = result
            print(f"\nTEST RUN LINK: {link}\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to upload test run: {e}\n")
        print("[DEBUG] sessionfinish is_confident:", is_confident())
        print("[DEBUG] sessionfinish test_run is None:", test_run is None)
        print(
            "[DEBUG] sessionfinish test_cases:",
            len(test_run.test_cases) if test_run else None,
        )
        print(
            "[DEBUG] sessionfinish conversational:",
            len(test_run.conversational_test_cases) if test_run else None,
        )
        print(
            "[DEBUG] sessionfinish identifier:",
            getattr(test_run, "identifier", None),
        )
