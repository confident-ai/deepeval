"""
Pytest configuration for LangGraph integration tests.

- Uploads traces directly to Confident AI Observatory (/v1/traces) after each test.
- Also creates a TestRun with test cases for the Test Runs UI.
- Each test case includes trace_uuid in additional_metadata for correlation.
- Test case fields are derived from trace_dict and test markers where available.

Field population sources (LLMApiTestCase schema from deepeval/test_run/api.py):
  - name: pytest nodeid
  - input: trace_dict["input"]["messages"][0]["content"] (first human message)
  - actual_output: trace_dict["output"]["messages"][-1]["content"] (last AI message)
  - expected_output: None (tests do not define expected outputs)
  - context: None (not a RAG application, no context provided)
  - retrieval_context: None (not a RAG application, no retriever)
  - tools_called: trace_dict["toolsCalled"] or trace_dict["toolSpans"]
  - expected_tools: None (tests do not define expected tools)
  - token_cost: sum of llmSpans[*].inputTokenCount + outputTokenCount (no cost rate)
  - completion_time: (endTime - startTime) in seconds from trace_dict timestamps
  - tags: trace_dict["tags"] (from CallbackHandler tags parameter)
  - additional_metadata: trace correlation + environment info
  - success: pytest test passed/failed
  - metricsData: None (no metrics evaluation)
  - trace: None (embedding causes 500 errors)
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
    """Extract a readable input string from trace_dict.

    Source: trace_dict["input"]["messages"][0]["content"]
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

    Source: trace_dict["output"]["messages"][-1]["content"] (last AI message)
    Prefers last AI message content if present, otherwise stringifies trace_dict["output"].
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
    trace_dict: Dict[str, Any],
) -> Optional[List[ToolCall]]:
    """Extract tools_called from trace_dict.

    Source: trace_dict["toolsCalled"] (preferred) or trace_dict["toolSpans"]
    Returns list of ToolCall objects or None if no tools were called.
    """
    result = []

    # First try top-level toolsCalled (most complete)
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


def _extract_expected_output(
    nodeid: str, item: pytest.Item, trace_dict: Dict[str, Any]
) -> Optional[str]:
    """Extract expected_output if test defines it.

    Source: pytest marker @pytest.mark.expected_output("...") or item attribute.

    IMPORTANT: We do NOT guess or fabricate expected_output.
    Current LangGraph tests do not define expected outputs (they only assert
    len(result["messages"]) > 0), so this returns None.
    """
    # Check for pytest marker
    marker = item.get_closest_marker("expected_output")
    if marker and marker.args:
        return _truncate(str(marker.args[0]))

    # Check for item attribute (e.g., set by fixture)
    if hasattr(item, "expected_output") and item.expected_output is not None:
        return _truncate(str(item.expected_output))

    # No expected output defined - return None (do not guess)
    return None


def _extract_expected_tools(
    nodeid: str, item: pytest.Item, trace_dict: Dict[str, Any]
) -> Optional[List[str]]:
    """Extract expected_tools if test defines them.

    Source: pytest marker @pytest.mark.expected_tools(["tool1", "tool2"]) or item attribute.

    IMPORTANT: We do NOT guess or fabricate expected_tools.
    Current LangGraph tests do not define expected tools, so this returns None.
    """
    # Check for pytest marker
    marker = item.get_closest_marker("expected_tools")
    if marker and marker.args:
        tools = marker.args[0]
        if isinstance(tools, list):
            return tools

    # Check for item attribute (e.g., set by fixture)
    if hasattr(item, "expected_tools") and item.expected_tools is not None:
        return item.expected_tools

    # No expected tools defined - return None (do not guess)
    return None


def _extract_context(
    nodeid: str, item: pytest.Item, trace_dict: Dict[str, Any]
) -> Optional[List[str]]:
    """Extract context if test defines it.

    Source: pytest marker @pytest.mark.context(["..."]) or item attribute.

    IMPORTANT: We do NOT guess or fabricate context.
    Current LangGraph tests are agent tests, not RAG - no context is provided.
    """
    # Check for pytest marker
    marker = item.get_closest_marker("context")
    if marker and marker.args:
        ctx = marker.args[0]
        if isinstance(ctx, list):
            return ctx

    # Check for item attribute
    if hasattr(item, "context") and item.context is not None:
        return item.context

    # No context defined - return None (do not guess)
    return None


def _extract_retrieval_context(
    nodeid: str, item: pytest.Item, trace_dict: Dict[str, Any]
) -> Optional[List[str]]:
    """Extract retrieval_context from trace if retriever was used.

    Source: trace_dict["retrieverSpans"] or pytest marker.

    IMPORTANT: We only populate this if actual retrieval happened.
    Current LangGraph tests do not use retrievers (retrieverSpans is empty).
    """
    # Check for pytest marker first
    marker = item.get_closest_marker("retrieval_context")
    if marker and marker.args:
        ctx = marker.args[0]
        if isinstance(ctx, list):
            return ctx

    # Check for item attribute
    if (
        hasattr(item, "retrieval_context")
        and item.retrieval_context is not None
    ):
        return item.retrieval_context

    # Check trace for retriever spans
    retriever_spans = trace_dict.get("retrieverSpans", [])
    if retriever_spans:
        # Extract retrieved documents from retriever spans
        contexts = []
        for span in retriever_spans:
            if isinstance(span, dict):
                output = span.get("output")
                if output:
                    # Retriever output is typically a list of documents
                    if isinstance(output, list):
                        for doc in output:
                            if isinstance(doc, dict):
                                content = doc.get("page_content") or doc.get(
                                    "content"
                                )
                                if content:
                                    contexts.append(_truncate(str(content)))
                            elif isinstance(doc, str):
                                contexts.append(_truncate(doc))
        if contexts:
            return contexts

    # No retrieval context - return None
    return None


def _extract_token_cost(trace_dict: Dict[str, Any]) -> Optional[float]:
    """Extract total token count from trace.

    Source: Sum of llmSpans[*].inputTokenCount + llmSpans[*].outputTokenCount

    NOTE: This returns total token COUNT, not dollar cost (we don't have pricing info).
    The field is named "token_cost" but we populate it with total tokens as a proxy.
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

    Source: trace_dict["tags"] (from CallbackHandler tags parameter)
            or pytest marker @pytest.mark.tags(["tag1", "tag2"])

    Returns None if no tags are defined.
    """
    tags = []

    # First, get tags from trace (from CallbackHandler)
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
    We set metricsData=None to signal "no metrics evaluated" (vs empty list
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

    # Extract all fields from trace_dict and test item
    input_str = _extract_input_from_trace(trace_dict)
    output_str = _extract_output_from_trace(trace_dict)
    tools_called = _extract_tools_called_from_trace(trace_dict)
    expected_output = _extract_expected_output(nodeid, item, trace_dict)
    expected_tools = _extract_expected_tools(nodeid, item, trace_dict)
    context = _extract_context(nodeid, item, trace_dict)
    retrieval_context = _extract_retrieval_context(nodeid, item, trace_dict)
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

    # Build LLMApiTestCase directly with camelCase field aliases.
    # We set metricsData=None (not []) to avoid the guard in update_test_run,
    # and trace=None to avoid server 500 errors when embedding traces.
    api_test_case = LLMApiTestCase(
        name=nodeid,
        input=input_str or f"LangGraph test: {test_name}",
        actualOutput=output_str or ("PASSED" if passed else "FAILED"),
        expectedOutput=expected_output,  # None unless test explicitly defines
        context=context,  # None - not a RAG app
        retrievalContext=retrieval_context,  # None - not a RAG app
        toolsCalled=tools_called,
        expectedTools=expected_tools,  # None unless test explicitly defines
        tokenCost=token_cost,  # Total token count from llmSpans
        completionTime=completion_time,  # Duration in seconds from timestamps
        tags=tags,  # From CallbackHandler tags
        additionalMetadata=additional_metadata,
        success=passed,
        metricsData=None,  # None = "no metrics evaluated" (bypasses guard)
        trace=None,  # Must be None - embedding traces causes 500s
        order=order,
        runDuration=completion_time or 0,  # Use completion_time as run duration
        evaluationCost=None,  # No evaluation performed
    )

    # Concise debug log showing which optional fields are populated
    print(
        f"[DEBUG] added api_test_case fields: "
        f"expectedOutput={expected_output is not None} "
        f"expectedTools={expected_tools is not None} "
        f"context={context is not None} "
        f"retrievalContext={retrieval_context is not None} "
        f"tokenCost={token_cost is not None} "
        f"completionTime={completion_time is not None} "
        f"tags={tags is not None}"
    )

    # Print values when present
    if token_cost is not None:
        print(f"[DEBUG]   tokenCost={token_cost:.1f} (total tokens)")
    if completion_time is not None:
        print(f"[DEBUG]   completionTime={completion_time:.3f}s")
    if tags:
        print(f"[DEBUG]   tags={tags}")

    # Directly add to test_run.test_cases, bypassing update_test_run guard
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
