/**
 * Canonical `confident.*` OTel attribute keys.
 *
 * Processors write these onto OTel spans and either the REST path or the
 * Confident AI collector reads them back, so a misspelling is invisible: the
 * write lands on a key nobody looks at and the field disappears from the trace
 * with no error.
 */
export const ConfidentAttr = {
  // Span-level fields, mirroring `updateCurrentSpan`.
  SPAN_CONTEXT: "confident.span.context",
  SPAN_EXPECTED_OUTPUT: "confident.span.expected_output",
  SPAN_EXPECTED_TOOLS: "confident.span.expected_tools",
  SPAN_INPUT: "confident.span.input",
  SPAN_METADATA: "confident.span.metadata",
  SPAN_METADATA_TOOL_ID: "confident.span.metadata.tool_id",
  SPAN_METRIC_COLLECTION: "confident.span.metric_collection",
  SPAN_NAME: "confident.span.name",
  SPAN_OUTPUT: "confident.span.output",
  SPAN_PROMPT_ALIAS: "confident.span.prompt_alias",
  SPAN_PROMPT_COMMIT_HASH: "confident.span.prompt_commit_hash",
  SPAN_PROMPT_LABEL: "confident.span.prompt_label",
  SPAN_PROMPT_VERSION: "confident.span.prompt_version",
  SPAN_RETRIEVAL_CONTEXT: "confident.span.retrieval_context",
  SPAN_TOOLS_CALLED: "confident.span.tools_called",
  SPAN_TYPE: "confident.span.type",

  // LLM span fields.
  LLM_COST_PER_INPUT_TOKEN: "confident.llm.cost_per_input_token",
  LLM_COST_PER_OUTPUT_TOKEN: "confident.llm.cost_per_output_token",
  LLM_INPUT_TOKEN_COUNT: "confident.llm.input_token_count",
  LLM_MODEL: "confident.llm.model",
  LLM_OUTPUT_TOKEN_COUNT: "confident.llm.output_token_count",

  // Agent span fields.
  AGENT_AGENT_HANDOFFS: "confident.agent.agent_handoffs",
  AGENT_AVAILABLE_TOOLS: "confident.agent.available_tools",

  // Tool span fields.
  TOOL_DESCRIPTION: "confident.tool.description",
  TOOL_NAME: "confident.tool.name",

  // Retriever span fields.
  RETRIEVER_CHUNK_SIZE: "confident.retriever.chunk_size",
  RETRIEVER_EMBEDDER: "confident.retriever.embedder",
  RETRIEVER_TOP_K: "confident.retriever.top_k",

  // Trace-level fields, mirroring `updateCurrentTrace`.
  TRACE_CONTEXT: "confident.trace.context",
  TRACE_ENVIRONMENT: "confident.trace.environment",
  TRACE_EXPECTED_OUTPUT: "confident.trace.expected_output",
  TRACE_EXPECTED_TOOLS: "confident.trace.expected_tools",
  TRACE_INPUT: "confident.trace.input",
  TRACE_METADATA: "confident.trace.metadata",
  TRACE_METRIC_COLLECTION: "confident.trace.metric_collection",
  TRACE_NAME: "confident.trace.name",
  TRACE_OUTPUT: "confident.trace.output",
  TRACE_RETRIEVAL_CONTEXT: "confident.trace.retrieval_context",
  TRACE_SESSION_ID: "confident.trace.session_id",
  TRACE_TAGS: "confident.trace.tags",
  TRACE_TEST_CASE_ID: "confident.trace.test_case_id",
  TRACE_THREAD_ID: "confident.trace.thread_id",
  TRACE_TOOLS_CALLED: "confident.trace.tools_called",
  TRACE_TURN_ID: "confident.trace.turn_id",
  TRACE_USER_ID: "confident.trace.user_id",

  // Plumbing that never leaves the SDK.
  INTERNAL_IS_AI_ROOT: "confident.internal.is_ai_root",
  INTERNAL_IS_OI_SPAN: "confident.internal.is_oi_span",
  INTERNAL_ROUTE_TO_REST: "confident.internal.route_to_rest",
  INTERNAL_TRACE_UUID: "confident.internal.trace_uuid",
} as const;
