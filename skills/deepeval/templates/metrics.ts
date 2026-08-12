import {
  AnswerRelevancyMetric,
  ContextualRelevancyMetric,
  StepEfficiencyMetric,
  TaskCompletionMetric,
} from "deepeval/metrics";

// Keep metrics in one module so eval files stay focused on app execution.
// Reuse existing project metrics and thresholds before adding new ones.
export const SINGLE_TURN_TRACE_METRICS = [
  new TaskCompletionMetric(),
  new StepEfficiencyMetric(),
];

export const SINGLE_TURN_NO_TRACING_METRICS = [new AnswerRelevancyMetric()];

export const MULTI_TURN_METRICS = [];

// Component-level metrics are span-specific. Do not create one shared
// COMPONENT_METRICS list for the whole app. Name each list after the exact
// component/span it evaluates, then attach it with either:
// - nextAgentSpan / nextLlmSpan / nextToolSpan / nextRetrieverSpan
// - observe({ metrics: [...] }) when the integration or manual instrumentation
//   creates the component span directly.
export const RETRIEVER_SPAN_METRICS = [new ContextualRelevancyMetric()];

export const GENERATOR_LLM_SPAN_METRICS = [new AnswerRelevancyMetric()];

export const TOOL_SPAN_METRICS = [];

export const PLANNER_AGENT_SPAN_METRICS = [];
