"""
Pytest configuration for LangChain integration tests.

Mirrors the LangGraph conftest.py structure for consistency.
"""

import os
import sys
import pytest
import datetime
from typing import Dict, Any, List, Optional
from dateutil import parser as dateutil_parser

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
    _test_run_identifier = f"langchain-integrations-{timestamp}"

    # Enable disk persistence and create the test run
    global_test_run_manager.save_to_disk = True
    global_test_run_manager.create_test_run(
        identifier=_test_run_identifier,
        file_name="tests/test_integrations/test_langchain",
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

    # 1) Upload trace directly to /v1/traces
    trace_uuid = _upload_trace_to_observatory(trace_dict)

    # 2) Add test case to TestRun
    if trace_uuid:
        _add_test_case_to_run(
            item, item.nodeid, report.passed, trace_uuid, trace_dict
        )


def _upload_trace_to_observatory(trace_dict: dict) -> str:
    """Upload trace dict directly to Confident AI Observatory via /v1/traces."""
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


# =============================================================================
# EXTRACTION HELPERS
# =============================================================================


def _truncate(s: str, max_len: int = MAX_FIELD_LENGTH) -> str:
    """Truncate string to max_len, adding ellipsis if truncated."""
    if s and len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _extract_input_from_trace(trace_dict: Dict[str, Any]) -> str:
    """Extract a readable input string from trace_dict."""
    trace_input = trace_dict.get("input")
    if trace_input is None:
        return ""

    if isinstance(trace_input, dict) and "messages" in trace_input:
        messages = trace_input.get("messages", [])
        if messages and isinstance(messages[0], dict):
            content = messages[0].get("content", "")
            if content:
                return _truncate(str(content))

    return _truncate(str(trace_input))


def _extract_output_from_trace(trace_dict: Dict[str, Any]) -> str:
    """Extract a readable output string from trace_dict."""
    trace_output = trace_dict.get("output")
    if trace_output is None:
        return ""

    if isinstance(trace_output, dict) and "messages" in trace_output:
        messages = trace_output.get("messages", [])
        if messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("type") == "ai":
                    content = msg.get("content", "")
                    if content:
                        return _truncate(str(content))
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
                if content:
                    return _truncate(str(content))

    return _truncate(str(trace_output))


def _extract_tools_called_from_trace(
    trace_dict: Dict[str, Any],
) -> Optional[List[ToolCall]]:
    """Extract tools_called from trace_dict."""
    result = []

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


def _extract_token_cost(trace_dict: Dict[str, Any]) -> Optional[float]:
    """Extract total token count from trace."""
    llm_spans = trace_dict.get("llmSpans", [])
    if not llm_spans:
        return None

    total_tokens = 0
    has_token_data = False

    for span in llm_spans:
        if not isinstance(span, dict):
            continue

        input_tokens = span.get("inputTokenCount")
        output_tokens = span.get("outputTokenCount")

        if input_tokens is not None:
            total_tokens += input_tokens
            has_token_data = True
        if output_tokens is not None:
            total_tokens += output_tokens
            has_token_data = True

    return float(total_tokens) if has_token_data else None


def _extract_completion_time(trace_dict: Dict[str, Any]) -> Optional[float]:
    """Extract completion time from trace timestamps."""
    start_time_str = trace_dict.get("startTime")
    end_time_str = trace_dict.get("endTime")

    if not start_time_str or not end_time_str:
        return None

    try:
        start_time = dateutil_parser.isoparse(start_time_str)
        end_time = dateutil_parser.isoparse(end_time_str)
        duration = (end_time - start_time).total_seconds()
        return duration if duration >= 0 else None
    except (ValueError, TypeError):
        return None


def _extract_tags(
    nodeid: str, item: pytest.Item, trace_dict: Dict[str, Any]
) -> Optional[List[str]]:
    """Extract tags from trace or test markers."""
    tags = []

    trace_tags = trace_dict.get("tags")
    if trace_tags and isinstance(trace_tags, list):
        tags.extend(trace_tags)

    marker = item.get_closest_marker("tags")
    if marker and marker.args:
        marker_tags = marker.args[0]
        if isinstance(marker_tags, list):
            tags.extend(marker_tags)

    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)

    return unique_tags if unique_tags else None


def _get_environment_info() -> Dict[str, str]:
    """Collect environment info for debugging."""
    info = {
        "python_version": sys.version.split()[0],
    }

    try:
        import langchain_core

        info["langchain_core_version"] = getattr(
            langchain_core, "__version__", "unknown"
        )
    except ImportError:
        pass

    return info


# =============================================================================
# TEST CASE CREATION
# =============================================================================


def _add_test_case_to_run(
    item: pytest.Item,
    nodeid: str,
    passed: bool,
    trace_uuid: str,
    trace_dict: Dict[str, Any],
):
    """Add a test case to the current TestRun."""
    from deepeval.test_run import global_test_run_manager
    from deepeval.test_run.api import LLMApiTestCase

    test_run = global_test_run_manager.test_run
    if test_run is None:
        return

    parts = nodeid.split("::")
    test_file = parts[0] if parts else nodeid
    test_name = parts[-1] if parts else nodeid

    input_str = _extract_input_from_trace(trace_dict)
    output_str = _extract_output_from_trace(trace_dict)
    tools_called = _extract_tools_called_from_trace(trace_dict)
    token_cost = _extract_token_cost(trace_dict)
    completion_time = _extract_completion_time(trace_dict)
    tags = _extract_tags(nodeid, item, trace_dict)

    additional_metadata = {
        "trace_uuid": trace_uuid,
        "pytest_nodeid": nodeid,
        "test_file": test_file,
        "test_name": test_name,
        "trace_name": trace_dict.get("name"),
        **_get_environment_info(),
    }

    order = len(test_run.test_cases)

    api_test_case = LLMApiTestCase(
        name=f"{nodeid} [{trace_uuid}]",
        input=input_str or f"LangChain test: {test_name}",
        actualOutput=output_str or ("PASSED" if passed else "FAILED"),
        expectedOutput=None,
        context=None,
        retrievalContext=None,
        toolsCalled=tools_called,
        expectedTools=None,
        tokenCost=token_cost,
        completionTime=completion_time,
        tags=tags,
        additionalMetadata=additional_metadata,
        success=passed,
        metricsData=None,
        trace=None,
        order=order,
        runDuration=completion_time or 0,
        evaluationCost=None,
    )

    print("[DEBUG] trace keys:", trace_dict.keys())
    print("[DEBUG] toolsCalled top-level:", bool(trace_dict.get("toolsCalled")))
    print("[DEBUG] toolSpans:", len(trace_dict.get("toolSpans", [])))
    print("[DEBUG] baseSpans:", len(trace_dict.get("baseSpans", [])))
    print(
        "[DEBUG] output:",
        type(trace_dict.get("output")),
        trace_dict.get("output"),
    )

    print(
        f"[DEBUG] added api_test_case fields: "
        f"tokenCost={token_cost is not None} "
        f"completionTime={completion_time is not None} "
        f"tags={tags is not None}"
    )

    if completion_time is not None:
        print(f"[DEBUG]   completionTime={completion_time:.3f}s")
    if tags:
        print(f"[DEBUG]   tags={tags}")

    test_run.add_test_case(api_test_case)
    print(
        f"[DEBUG] after add_test_case, test_cases: {len(test_run.test_cases)}"
    )


# =============================================================================
# SESSION FINISH
# =============================================================================


def pytest_sessionfinish(session: pytest.Session, exitstatus):
    """Upload the TestRun at the end of the session."""
    from deepeval.confident.api import is_confident
    from deepeval.test_run import global_test_run_manager

    print("Running teardown with pytest sessionfinish...")

    if not is_confident():
        return

    test_run = global_test_run_manager.test_run
    if test_run is None:
        print("[DEBUG] sessionfinish: test_run is None, skipping upload")
        return

    if (
        len(test_run.test_cases) == 0
        and len(test_run.conversational_test_cases) == 0
    ):
        print("[DEBUG] sessionfinish: no test cases found, skipping upload")
        return

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
