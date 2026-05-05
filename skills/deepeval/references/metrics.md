# Metrics

Use 3-5 metrics for the first eval suite when the user is unsure. More metrics
make iteration slower and harder to interpret. Reuse existing project metrics
and thresholds before adding new ones.

## Required Rule

Single-turn `LLMTestCase` evals must use single-turn metrics.

Multi-turn `ConversationalTestCase` evals must use multi-turn conversational
metrics. Do not use `AnswerRelevancyMetric`, `FaithfulnessMetric`, or other
single-turn `LLMTestCase` metrics on multi-turn end-to-end evals.

## Metric Types

Choose metrics by what the user wants to measure, not only by app type.

| Type | Use when | Examples |
| --- | --- | --- |
| Custom criteria | The success criteria is product- or domain-specific | `GEval`, `DAGMetric`, `ConversationalGEval`, `ConversationalDAGMetric` |
| RAG retriever | You need to evaluate retrieved context quality | `ContextualRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric` |
| RAG generator | You need to evaluate the final answer against context | `AnswerRelevancyMetric`, `FaithfulnessMetric` |
| Agentic flow | You need to evaluate task completion, plans, steps, tools, or arguments | `TaskCompletionMetric`, `ToolCorrectnessMetric`, `ArgumentCorrectnessMetric`, `PlanAdherenceMetric`, `PlanQualityMetric`, `StepEfficiencyMetric` |
| Multi-turn chatbot | You need to evaluate an entire conversation | `ConversationCompletenessMetric`, `RoleAdherenceMetric`, `TurnRelevancyMetric`, `ConversationalGEval` |
| Safety and compliance | You need to detect risky or policy-violating outputs | `BiasMetric`, `ToxicityMetric`, `PIILeakageMetric`, `MisuseMetric`, `RoleViolationMetric`, `NonAdviceMetric` |
| Format / structure | You need output to match a schema or instruction set | `JsonCorrectnessMetric`, `PromptAlignmentMetric` |
| Other task-specific quality | The app is summarization, hallucination-sensitive, image-based, or otherwise specialized | `SummarizationMetric`, `HallucinationMetric`, multimodal metrics |

Aim to include at least one custom metric when the user's definition of success
is not fully captured by a predefined metric. In practice, custom metrics should
usually be `GEval` for single-turn evals or `ConversationalGEval` for multi-turn
evals.

## Default If User Is Unsure

If the user says "I don't know" or gives no metric preference:

- Use 3-5 metrics.
- Put metrics on the end-to-end eval first.
- Do not add safety metrics by default unless the app is safety/compliance
  sensitive or the user asks for them.
- Use about half custom metrics and half system-specific metrics.
- Add component-level metrics only after E2E/traces show component failures, or
  if the user explicitly wants component evals.

Good system-specific defaults:

- Agent: `TaskCompletionMetric` plus tool/argument correctness only when
  `tools_called` data exists.
- RAG: `FaithfulnessMetric`, `AnswerRelevancyMetric`, and
  `ContextualRelevancyMetric` are strong candidates.
- Multi-turn chatbot: use conversational metrics only, plus a
  `ConversationalGEval` custom criterion when product-specific behavior matters.

For custom metrics, assume `GEval` for single-turn or `ConversationalGEval` for
multi-turn. There is a very high chance this is the right custom metric type.
Do not start with DAG unless the user already has a DAG metric or specifically
needs decision-tree scoring.

Use `GEval` when scoring is subjective or there is no predefined metric for the
thing the user cares about. Correctness is a common example: there is no generic
"correctness metric" because correctness depends on the task. Define a `GEval`
named `Correctness` and write criteria that explain what correct means for this
app.

Use `DAGMetric` only when the metric is decision-based: the score should follow
explicit branches, checks, or deterministic rubric paths. DAG is useful when the
metric is more like a decision tree than a subjective judge. Do not start with
DAG for ordinary subjective scoring.

When choosing `GEval.evaluation_params`, include only fields the test case will
actually have. Be especially careful with reference-space params like
`expected_output`, `context`, `retrieval_context`, or `expected_tools`; if the
dataset or app does not provide them, the metric will fail at runtime. Prefer
`input` and `actual_output` unless the eval plan explicitly creates the
reference fields.

If existing project metrics are present, use them first. If there are too many,
tell the user: "You already have a lot of metrics here, which may make evals
slow or hard to interpret. I recommend narrowing the first run to the highest
signal metrics."

## Reference-Based Metrics

Some metrics require reference fields. Use them sparingly unless the plan
includes those expected values, because missing fields will cause metric errors.

Reference-based fields include:

- `expected_output`
- `expected_outcome`
- `expected_tools`
- `context`
- `retrieval_context`

Examples:

- `ContextualPrecisionMetric` and `ContextualRecallMetric` need
  `expected_output`.
- `ToolCorrectnessMetric` needs `expected_tools`.
- Multi-turn outcome metrics may depend on `expected_outcome`.
- RAG grounding metrics need `retrieval_context`.

If the dataset does not include the required fields, choose metrics that match
available fields or update the dataset generation/loading plan first.

## Common Single-Turn Metrics

| Metric | What it checks | Required test case fields |
| --- | --- | --- |
| `AnswerRelevancyMetric` | Output answers the input | `input`, `actual_output` |
| `FaithfulnessMetric` | Output is grounded in retrieved context | `input`, `actual_output`, `retrieval_context` |
| `ContextualRelevancyMetric` | Retrieved context is relevant to input | `input`, `retrieval_context` |
| `ContextualPrecisionMetric` | Relevant context is ranked highly | `input`, `retrieval_context`, `expected_output` |
| `ContextualRecallMetric` | Retrieved context covers expected answer | `input`, `retrieval_context`, `expected_output` |
| `TaskCompletionMetric` | Agent/app completed the task | `input`, `actual_output` |
| `ToolCorrectnessMetric` | Called tools match expected tools | `input`, `tools_called`, `expected_tools` |
| `ArgumentCorrectnessMetric` | Tool arguments are correct | `input`, `tools_called` |
| `JsonCorrectnessMetric` | Output matches expected schema | `input`, `actual_output`; constructor needs `expected_schema` |
| `PromptAlignmentMetric` | Output follows prompt instructions | `input`, `actual_output`; constructor needs `prompt_instructions` |
| `GEval` | Custom single-turn criteria | constructor needs `name`, `criteria` or `evaluation_steps`, and `evaluation_params` |

## Common Multi-Turn Metrics

| Metric | What it checks | Required test case fields |
| --- | --- | --- |
| `ConversationCompletenessMetric` | Conversation achieved the expected outcome | `turns` with `role`, `content` |
| `RoleAdherenceMetric` | Assistant stayed in role across turns | `turns` with `role`, `content` |
| `TurnRelevancyMetric` | Assistant turns are relevant | `turns` with `role`, `content` |
| `TurnFaithfulnessMetric` | Turns are faithful to retrieval context | `turns` with `role`, `content`, `retrieval_context` |
| `TurnContextualRelevancyMetric` | Turn retrieval context is relevant | `turns` with `role`, `content`, retrieval context |
| `GoalAccuracyMetric` | Conversation achieved the user's goal | `turns` with `role`, `content` |
| `TopicAdherenceMetric` | Conversation stayed on allowed topics | `turns` with `role`, `content`; constructor needs `relevant_topics` |
| `ConversationalGEval` | Custom multi-turn criteria | constructor needs `name` and `criteria` or `evaluation_steps` |

## Choosing Metrics

Ask what the user cares about in product terms first. Then map that to metrics.

Ask:

- What failure would be unacceptable in production?
- Is success about final answer quality, retrieved context, tool use, safety,
  conversation completion, or output format?
- Do we need a custom criterion because the product definition of "good" is
  domain-specific?
- Which fields does the dataset/test case actually contain?

Mappings:

- "Does it answer correctly?" -> `AnswerRelevancyMetric` or task-specific `GEval`
- "Is it grounded in docs?" -> `FaithfulnessMetric` plus contextual metrics
- "Did the agent finish the task?" -> `TaskCompletionMetric`
- "Did it use the right tool?" -> `ToolCorrectnessMetric`
- "Did the chatbot complete the conversation?" -> `ConversationCompletenessMetric`
- "Did it stay in character?" -> `RoleAdherenceMetric`

If unsure, start with 3-5 E2E metrics and add component-level metrics only after
the first run reveals where the app is failing.
