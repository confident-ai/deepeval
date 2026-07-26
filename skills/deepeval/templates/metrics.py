from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
)

# Keep metrics in one module so eval files stay focused on app execution.
# Reuse existing project metrics and thresholds before adding new ones.
SINGLE_TURN_TRACE_METRICS = [
    TaskCompletionMetric(),
    StepEfficiencyMetric(),
]

SINGLE_TURN_NO_TRACING_METRICS = [
    AnswerRelevancyMetric(),
]

MULTI_TURN_METRICS = []

# Component-level metrics are span-specific. Do not create one shared
# COMPONENT_METRICS list for the whole app. Name each list after the exact
# component/span it evaluates, then attach it with either:
# - next_agent_span / next_llm_span / next_tool_span / next_retriever_span
# - @observe(metrics=[...]) when the integration or manual instrumentation
#   creates the component span directly.
RETRIEVER_SPAN_METRICS = [
    ContextualRelevancyMetric(),
]

GENERATOR_LLM_SPAN_METRICS = [
    AnswerRelevancyMetric(),
]

TOOL_SPAN_METRICS = []

PLANNER_AGENT_SPAN_METRICS = []
