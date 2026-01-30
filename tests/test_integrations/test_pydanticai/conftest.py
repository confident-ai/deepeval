"""
Pytest configuration for PydanticAI integration tests.

- Uploads traces directly to Confident AI Observatory (/v1/traces) after each test.
- Also creates a TestRun with test cases for the Test Runs UI.
- Each test case includes trace_uuid in additional_metadata for correlation.
- Test case fields are derived from trace_dict and test markers where available.

Field population sources (LLMApiTestCase schema from deepeval/test_run/api.py):
  - name: pytest nodeid
  - input: trace_dict["input"] (system + user messages)
  - actual_output: trace_dict["output"] (agent response)
  - expected_output: None (tests do not define expected outputs)
  - context: None (not a RAG application, no context provided)
  - retrieval_context: None (not a RAG application, no retriever)
  - tools_called: trace_dict["toolSpans"]
  - expected_tools: None (tests do not define expected tools)
  - token_cost: sum of llmSpans[*].inputTokenCount + outputTokenCount (no cost rate)
  - completion_time: (endTime - startTime) in seconds from trace_dict timestamps
  - tags: trace_dict["tags"] (from ConfidentInstrumentationSettings tags parameter)
  - additional_metadata: trace correlation + environment info
  - success: pytest test passed/failed
  - metricsData: None (no metrics evaluation)
  - trace: None (embedding causes 500 errors)
"""

import os
import sys
import pytest
import datetime
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dateutil import parser as dateutil_parser

from deepeval.test_case import ToolCall


_logger = logging.getLogger(__name__)

# Module-level state for TestRun
_test_run_identifier = None

# Max length for input/output strings to avoid large payloads
MAX_FIELD_LENGTH = 2000


def _upload_enabled() -> bool:
    """Check if test run uploads are enabled via INTEGRATION_TESTS_UPLOAD_TEST_RUNS env var.

    Returns True only if the env var is set to a truthy value ("1", "true", "yes").
    Default is OFF (False) - no uploads, no network calls, no credentials needed.
    """
    val = (
        os.environ.get("INTEGRATION_TESTS_UPLOAD_TEST_RUNS", "").lower().strip()
    )
    return val in ("1", "true", "yes")


# =============================================================================
# ISOLATION FIXTURE (preserves original behavior)
# =============================================================================


@pytest.fixture(autouse=True)
def deepeval_isolated_no_disk(tmp_path, monkeypatch):
    """Isolate test from disk writes and uploads."""
    hidden = tmp_path / ".deepeval"
    hidden.mkdir(parents=True, exist_ok=True)

    # import the modules we need to patch
    import deepeval.constants as consts
    import deepeval.key_handler as keyh
    import deepeval.test_run.test_run as tr
    import deepeval.dataset.dataset as ds

    # point both constants modules at our isolated dir
    monkeypatch.setattr(consts, "HIDDEN_DIR", str(hidden), raising=False)
    monkeypatch.setattr(keyh, "HIDDEN_DIR", str(hidden), raising=False)

    tmp_temp = hidden / ".temp_test_run_data.json"
    tmp_latest = hidden / ".latest_test_run.json"

    # patch both modules that reference these file paths:
    for mod in (tr, ds):
        monkeypatch.setattr(mod, "TEMP_FILE_PATH", str(tmp_temp), raising=False)
        monkeypatch.setattr(
            mod, "LATEST_TEST_RUN_FILE_PATH", str(tmp_latest), raising=False
        )

    # make sure the manager uses our temp file path,
    # and disable writes and uploads
    tr.global_test_run_manager.temp_file_path = str(tmp_temp)
    tr.global_test_run_manager.save_to_disk = False
    tr.global_test_run_manager.disable_request = True

    # at the class level ensure no disk writing methods so a plugin
    # or code path can't write anyway.
    monkeypatch.setattr(
        tr.TestRunManager,
        "save_test_run",
        lambda self, *a, **k: None,
        raising=False,
    )
    monkeypatch.setattr(
        tr.TestRunManager,
        "save_final_test_run_link",
        lambda self, *a, **k: None,
        raising=False,
    )
    monkeypatch.setattr(
        tr.TestRunManager,
        "save_test_run_locally",
        lambda self: None,
        raising=False,
    )

    # ensure the dir exists before portalocker could be touched by anything else
    hidden.mkdir(parents=True, exist_ok=True)

    yield


# =============================================================================
# PYTEST HOOKS FOR TRACE UPLOAD
# =============================================================================


def pytest_configure(config):
    """Set environment variables needed for upload."""
    os.environ["CONFIDENT_OPEN_BROWSER"] = "0"
    os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = "1"


def pytest_sessionstart(session: pytest.Session):
    """Create a TestRun at the start of the pytest session."""
    if not _upload_enabled():
        return

    from deepeval.confident.api import is_confident

    if not is_confident():
        return

    from deepeval.test_run import global_test_run_manager

    global _test_run_identifier

    # Create a unique identifier for this test run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _test_run_identifier = f"pydanticai-integrations-{timestamp}"

    # Enable disk persistence and create the test run
    global_test_run_manager.save_to_disk = True
    global_test_run_manager.create_test_run(
        identifier=_test_run_identifier,
        file_name="tests/test_integrations/test_pydanticai",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call):
    """After each test call phase, upload trace and add test case to TestRun."""
    outcome = yield
    report = outcome.get_result()

    # Only process after the test call phase (not setup/teardown)
    if call.when != "call":
        return

    if not _upload_enabled():
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

    # 2) Add test case to TestRun with data extracted from trace_dict
    if trace_uuid:
        _add_test_case_to_run(
            item, item.nodeid, report.passed, trace_uuid, trace_dict
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
        _logger.debug("UPLOADED TRACE UUID: %s", trace_uuid)
        return trace_uuid
    except Exception:
        _logger.exception("Failed to upload trace %s", trace_uuid)
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
    """Extract a readable input string from trace_dict.

    PydanticAI traces have input as list of messages:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    """
    trace_input = trace_dict.get("input")
    if trace_input is None:
        return ""

    # PydanticAI format: input is a list of message dicts
    if isinstance(trace_input, list):
        # Find user message
        for msg in trace_input:
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    return _truncate(str(content))
        # Fallback to last message
        if trace_input:
            last = trace_input[-1]
            if isinstance(last, dict):
                return _truncate(str(last.get("content", "")))

    # Fallback: stringify the input
    return _truncate(str(trace_input))


def _extract_output_from_trace(trace_dict: Dict[str, Any]) -> str:
    """Extract a readable output string from trace_dict.

    PydanticAI traces have output as a string (the agent's response).
    """
    trace_output = trace_dict.get("output")
    if trace_output is None:
        return ""

    # PydanticAI format: output is typically a string
    if isinstance(trace_output, str):
        return _truncate(trace_output)

    # Fallback: stringify the output
    return _truncate(str(trace_output))


def _extract_tools_called_from_trace(
    trace_dict: Dict[str, Any],
) -> Optional[List[ToolCall]]:
    """Extract tools_called from trace_dict.

    Source: trace_dict["toolSpans"]
    Returns list of ToolCall objects or None if no tools were called.
    """
    result = []

    # Try toolSpans
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
                            tool_input if isinstance(tool_input, dict) else None
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
    """Extract total token count from trace.

    Source: Sum of llmSpans[*].inputTokenCount + llmSpans[*].outputTokenCount

    NOTE: This returns total token COUNT, not dollar cost (we don't have pricing info).
    Returns None if no token info is available.
    """
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
    """Extract completion time (duration) from trace timestamps.

    Source: (trace_dict["endTime"] - trace_dict["startTime"]) in seconds

    Returns None if timestamps are missing or invalid.
    """
    start_time_str = trace_dict.get("startTime")
    end_time_str = trace_dict.get("endTime")

    if not start_time_str or not end_time_str:
        return None

    try:
        # Parse ISO 8601 timestamps
        start_time = dateutil_parser.isoparse(start_time_str)
        end_time = dateutil_parser.isoparse(end_time_str)

        # Calculate duration in seconds
        duration = (end_time - start_time).total_seconds()
        return duration if duration >= 0 else None
    except (ValueError, TypeError):
        return None


def _extract_tags(
    nodeid: str, item: pytest.Item, trace_dict: Dict[str, Any]
) -> Optional[List[str]]:
    """Extract tags from trace or test markers.

    Source: trace_dict["tags"] (from ConfidentInstrumentationSettings tags parameter)
            or pytest marker @pytest.mark.tags(["tag1", "tag2"])

    Returns None if no tags are defined.
    """
    tags = []

    # First, get tags from trace
    trace_tags = trace_dict.get("tags")
    if trace_tags and isinstance(trace_tags, list):
        tags.extend(trace_tags)

    # Check for pytest marker to add additional tags
    marker = item.get_closest_marker("tags")
    if marker and marker.args:
        marker_tags = marker.args[0]
        if isinstance(marker_tags, list):
            tags.extend(marker_tags)

    # Deduplicate while preserving order
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

    # Try to get pydantic_ai version
    try:
        import pydantic_ai

        info["pydantic_ai_version"] = getattr(
            pydantic_ai, "__version__", "unknown"
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
    """Add a test case to the current TestRun with data extracted from trace_dict."""
    from deepeval.test_run import global_test_run_manager
    from deepeval.test_run.api import LLMApiTestCase

    test_run = global_test_run_manager.test_run
    if test_run is None:
        return

    # Parse nodeid for metadata
    parts = nodeid.split("::")
    test_file = parts[0] if parts else nodeid
    test_name = parts[-1] if parts else nodeid

    # Extract all fields from trace_dict
    input_str = _extract_input_from_trace(trace_dict)
    output_str = _extract_output_from_trace(trace_dict)
    tools_called = _extract_tools_called_from_trace(trace_dict)
    token_cost = _extract_token_cost(trace_dict)
    completion_time = _extract_completion_time(trace_dict)
    tags = _extract_tags(nodeid, item, trace_dict)

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

    # Build LLMApiTestCase
    api_test_case = LLMApiTestCase(
        name=f"{nodeid} [{trace_uuid}]",
        input=input_str or f"PydanticAI test: {test_name}",
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

    _logger.debug("[DEBUG] trace keys: %s", list(trace_dict.keys()))
    _logger.debug("[DEBUG] toolSpans: %d", len(trace_dict.get("toolSpans", [])))
    _logger.debug(
        "[DEBUG] output: %s %s",
        type(trace_dict.get("output")),
        trace_dict.get("output"),
    )

    _logger.debug(
        "[DEBUG] added api_test_case fields: tokenCost=%s completionTime=%s tags=%s",
        token_cost is not None,
        completion_time is not None,
        tags is not None,
    )

    if completion_time is not None:
        _logger.debug("[DEBUG]   completionTime=%.3fs", completion_time)
    if tags:
        _logger.debug("[DEBUG]   tags=%s", tags)

    test_run.add_test_case(api_test_case)
    _logger.debug(
        "[DEBUG] after add_test_case, test_cases: %d", len(test_run.test_cases)
    )


# =============================================================================
# SESSION FINISH
# =============================================================================


def pytest_sessionfinish(session: pytest.Session, exitstatus):
    """Upload the TestRun at the end of the session."""

    if not _upload_enabled():
        return

    _logger.debug("Running teardown with pytest sessionfinish...")

    from deepeval.confident.api import is_confident
    from deepeval.test_run import global_test_run_manager

    if not is_confident():
        return

    test_run = global_test_run_manager.test_run
    if test_run is None:
        _logger.debug(
            "[DEBUG] sessionfinish: test_run is None, skipping upload"
        )
        return

    if (
        len(test_run.test_cases) == 0
        and len(test_run.conversational_test_cases) == 0
    ):
        _logger.debug(
            "[DEBUG] sessionfinish: no test cases found, skipping upload"
        )
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
            _logger.debug("TEST RUN LINK: %s", link)
    except Exception:
        _logger.exception("Failed to upload test run")
