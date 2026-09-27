"""Canonical ``confident.*`` OTel attribute keys.

Integrations write these onto OTel spans and ``ConfidentSpanExporter`` reads
them back, so a misspelling is invisible: the write lands on a key nobody
looks at and the field disappears from the trace with no error.
"""


class ConfidentAttr:
    # Span-level fields, mirroring ``update_current_span``.
    SPAN_CONTEXT = "confident.span.context"
    SPAN_EXPECTED_OUTPUT = "confident.span.expected_output"
    SPAN_EXPECTED_TOOLS = "confident.span.expected_tools"
    SPAN_INPUT = "confident.span.input"
    SPAN_INTEGRATION = "confident.span.integration"
    SPAN_METADATA = "confident.span.metadata"
    SPAN_METRIC_COLLECTION = "confident.span.metric_collection"
    SPAN_NAME = "confident.span.name"
    SPAN_OUTPUT = "confident.span.output"
    SPAN_PARENT_UUID = "confident.span.parent_uuid"
    SPAN_PROMPT = "confident.span.prompt"
    SPAN_PROMPT_ALIAS = "confident.span.prompt_alias"
    SPAN_PROMPT_COMMIT_HASH = "confident.span.prompt_commit_hash"
    SPAN_PROMPT_LABEL = "confident.span.prompt_label"
    SPAN_PROMPT_VERSION = "confident.span.prompt_version"
    SPAN_PROVIDER = "confident.span.provider"
    SPAN_RETRIEVAL_CONTEXT = "confident.span.retrieval_context"
    SPAN_TOOLS_CALLED = "confident.span.tools_called"
    SPAN_TYPE = "confident.span.type"

    # LLM span fields.
    LLM_COST_PER_INPUT_TOKEN = "confident.llm.cost_per_input_token"
    LLM_COST_PER_OUTPUT_TOKEN = "confident.llm.cost_per_output_token"
    LLM_INPUT_TOKEN_COUNT = "confident.llm.input_token_count"
    LLM_MODEL = "confident.llm.model"
    LLM_OUTPUT_TOKEN_COUNT = "confident.llm.output_token_count"
    LLM_PROMPT = "confident.llm.prompt"

    # Agent span fields.
    AGENT_AGENT_HANDOFFS = "confident.agent.agent_handoffs"
    AGENT_AVAILABLE_TOOLS = "confident.agent.available_tools"
    AGENT_NAME = "confident.agent.name"
    AGENT_TOOL_DEFINITIONS = "confident.agent.tool_definitions"

    # Tool span fields.
    TOOL_DESCRIPTION = "confident.tool.description"
    TOOL_NAME = "confident.tool.name"

    # Retriever span fields.
    RETRIEVER_CHUNK_SIZE = "confident.retriever.chunk_size"
    RETRIEVER_EMBEDDER = "confident.retriever.embedder"
    RETRIEVER_TOP_K = "confident.retriever.top_k"

    # Trace-level fields, mirroring ``update_current_trace``.
    TRACE_CONTEXT = "confident.trace.context"
    TRACE_ENVIRONMENT = "confident.trace.environment"
    TRACE_EXPECTED_TOOLS = "confident.trace.expected_tools"
    TRACE_INPUT = "confident.trace.input"
    TRACE_LLM_TEST_CASE_ACTUAL_OUTPUT = (
        "confident.trace.llm_test_case.actual_output"
    )
    TRACE_LLM_TEST_CASE_CONTEXT = "confident.trace.llm_test_case.context"
    TRACE_LLM_TEST_CASE_EXPECTED_OUTPUT = (
        "confident.trace.llm_test_case.expected_output"
    )
    TRACE_LLM_TEST_CASE_EXPECTED_TOOLS = (
        "confident.trace.llm_test_case.expected_tools"
    )
    TRACE_LLM_TEST_CASE_INPUT = "confident.trace.llm_test_case.input"
    TRACE_LLM_TEST_CASE_RETRIEVAL_CONTEXT = (
        "confident.trace.llm_test_case.retrieval_context"
    )
    TRACE_LLM_TEST_CASE_TOOLS_CALLED = (
        "confident.trace.llm_test_case.tools_called"
    )
    TRACE_METADATA = "confident.trace.metadata"
    TRACE_METRIC_COLLECTION = "confident.trace.metric_collection"
    TRACE_NAME = "confident.trace.name"
    TRACE_OUTPUT = "confident.trace.output"
    TRACE_RETRIEVAL_CONTEXT = "confident.trace.retrieval_context"
    TRACE_TAGS = "confident.trace.tags"
    TRACE_TEST_CASE_ID = "confident.trace.test_case_id"
    TRACE_TEST_RUN_ID = "confident.trace.test_run_id"
    TRACE_THREAD_ID = "confident.trace.thread_id"
    TRACE_TOOLS_CALLED = "confident.trace.tools_called"
    TRACE_TURN_ID = "confident.trace.turn_id"
    TRACE_USER_ID = "confident.trace.user_id"
