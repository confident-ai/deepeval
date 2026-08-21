<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/confident-ai/deepeval/main/assets/hero/wordmark-dark.svg">
        <img alt="DeepEval." src="https://raw.githubusercontent.com/confident-ai/deepeval/main/assets/hero/wordmark-light.svg" width="520">
    </picture>
</p>

<p align="center">
    <h1 align="center">The LLM Evaluation Framework for TypeScript</h1>
</p>

<h4 align="center">
    <p>
        <a href="https://deepeval.com/docs/getting-started?utm_source=GitHub">Documentation</a> |
        <a href="#quickstart">Quickstart</a> |
        <a href="#metrics">Metrics</a> |
        <a href="#tracing-and-integrations">Integrations</a> |
        <a href="https://github.com/confident-ai/deepeval">Python SDK</a>
    <p>
</h4>

**DeepEval** is an open-source LLM evaluation framework. It's similar to Vitest — in fact it runs _as_ Vitest — but specialized for testing LLM apps, with research-backed metrics like G-Eval, task completion, answer relevancy, and hallucination that use LLM-as-a-judge and run **locally on your machine**.

```typescript
await expect(testCase).toPass([correctnessMetric]);
```

Whether you're building agents, RAG pipelines, or chatbots with LangChain, Mastra, the AI SDK, or OpenAI, you can evaluate your app **end-to-end** as a black box, over **complete agent trajectories**, or at **individual steps** like LLM calls, tool use, and retrieval.

This package is the TypeScript SDK. Python remains DeepEval's most complete implementation — see [Python vs TypeScript](#python-vs-typescript) for the gaps.

<br />

# Quickstart

```bash
npm install --save-dev deepeval
```

Log in to [Confident AI](https://www.confident-ai.com?utm_source=deepeval&utm_medium=github&utm_content=quickstart_login&ref_page=github/typescript-readme) to keep evaluation results on the cloud and compare runs over time. It's free and takes no extra code, but it's optional — evals print to your terminal either way.

```bash
npx deepeval login
```

Then write a test file. Metrics judge with OpenAI by default, so set `OPENAI_API_KEY` first (`.env.local` and `.env` are auto-loaded).

```typescript
import { LLMTestCase, SingleTurnParams } from "deepeval/test-case";
import { GEval } from "deepeval/metrics";
import { it, expect } from "vitest";
import "deepeval/vitest";

it("gives a correct answer", async () => {
  const correctnessMetric = new GEval({
    name: "Correctness",
    criteria:
      "Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluationParams: [
      SingleTurnParams.ACTUAL_OUTPUT,
      SingleTurnParams.EXPECTED_OUTPUT,
    ],
    threshold: 0.5,
  });
  const testCase = new LLMTestCase({
    input: "What if these shoes don't fit?",
    // Replace this with the actual output from your LLM application
    actualOutput: "You have 30 days to get a full refund at no extra cost.",
    expectedOutput: "We offer a 30-day full refund at no extra costs.",
  });

  await expect(testCase).toPass([correctnessMetric]);
});
```

```bash
npx deepeval test run example.test.ts
```

Scores range from 0 to 1, and `threshold` decides whether the test passes. Every metric also explains itself, so a failure tells you _why_ it failed. [Read the docs](https://deepeval.com/docs/getting-started?utm_source=GitHub) for the full walkthrough.

<details>
<summary><b>Running under plain <code>vitest</code> instead of the CLI</b></summary>

Importing `deepeval/vitest` registers the `toPass()` matcher. `npx deepeval test run` also injects the matcher and the test-run reporter for you; to get the same from your own `vitest` command, register them in your config:

```typescript
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    setupFiles: ["deepeval/vitest"],
    globalSetup: ["deepeval/vitest/global-setup"],
    testTimeout: 120_000,
    hookTimeout: 120_000,
  },
});
```

</details>

<br />

# Metrics

Every metric is a class from `deepeval/metrics`, takes an options object, and returns a score with a reason. Full list and parameters in the [metrics docs](https://deepeval.com/docs/metrics-introduction?utm_source=GitHub).

**Custom, all-purpose:** [`GEval`](https://deepeval.com/docs/metrics-llm-evals) evaluates against any criteria you write in plain English; [`DAGMetric`](https://deepeval.com/docs/metrics-dag) builds a deterministic decision tree of LLM judgements when you need repeatable verdicts.

<details>
<summary><b>Agentic</b></summary>

[`TaskCompletionMetric`](https://deepeval.com/docs/metrics-task-completion), [`ToolCorrectnessMetric`](https://deepeval.com/docs/metrics-tool-correctness), [`GoalAccuracyMetric`](https://deepeval.com/docs/metrics-goal-accuracy), [`StepEfficiencyMetric`](https://deepeval.com/docs/metrics-step-efficiency), [`PlanAdherenceMetric`](https://deepeval.com/docs/metrics-plan-adherence), [`PlanQualityMetric`](https://deepeval.com/docs/metrics-plan-quality), [`ToolUseMetric`](https://deepeval.com/docs/metrics-tool-use), [`ArgumentCorrectnessMetric`](https://deepeval.com/docs/metrics-argument-correctness)

</details>

<details>
<summary><b>RAG</b></summary>

[`AnswerRelevancyMetric`](https://deepeval.com/docs/metrics-answer-relevancy), [`FaithfulnessMetric`](https://deepeval.com/docs/metrics-faithfulness), [`ContextualRecallMetric`](https://deepeval.com/docs/metrics-contextual-recall), [`ContextualPrecisionMetric`](https://deepeval.com/docs/metrics-contextual-precision), [`ContextualRelevancyMetric`](https://deepeval.com/docs/metrics-contextual-relevancy)

</details>

<details>
<summary><b>Multi-turn</b></summary>

[`KnowledgeRetentionMetric`](https://deepeval.com/docs/metrics-knowledge-retention), [`ConversationCompletenessMetric`](https://deepeval.com/docs/metrics-conversation-completeness), [`TurnRelevancyMetric`](https://deepeval.com/docs/metrics-turn-relevancy), [`TurnFaithfulnessMetric`](https://deepeval.com/docs/metrics-turn-faithfulness), [`RoleAdherenceMetric`](https://deepeval.com/docs/metrics-role-adherence), `TopicAdherenceMetric`, `TurnContextualPrecisionMetric`, `TurnContextualRecallMetric`, `TurnContextualRelevancyMetric`, `ConversationalGEval`, `ConversationalDAGMetric`

</details>

<details>
<summary><b>MCP</b></summary>

[`MCPTaskCompletionMetric`](https://deepeval.com/docs/metrics-mcp-task-completion), [`MCPUseMetric`](https://deepeval.com/docs/metrics-mcp-use), [`MultiTurnMCPUseMetric`](https://deepeval.com/docs/metrics-multi-turn-mcp-use)

</details>

<details>
<summary><b>Multimodal</b></summary>

[`TextToImageMetric`](https://deepeval.com/docs/multimodal-metrics-text-to-image), [`ImageEditingMetric`](https://deepeval.com/docs/multimodal-metrics-image-editing), [`ImageCoherenceMetric`](https://deepeval.com/docs/multimodal-metrics-image-coherence), [`ImageHelpfulnessMetric`](https://deepeval.com/docs/multimodal-metrics-image-helpfulness), [`ImageReferenceMetric`](https://deepeval.com/docs/multimodal-metrics-image-reference)

</details>

<details>
<summary><b>Safety, correctness and deterministic</b></summary>

[`HallucinationMetric`](https://deepeval.com/docs/metrics-hallucination), [`SummarizationMetric`](https://deepeval.com/docs/metrics-summarization), [`BiasMetric`](https://deepeval.com/docs/metrics-bias), [`ToxicityMetric`](https://deepeval.com/docs/metrics-toxicity), [`JsonCorrectnessMetric`](https://deepeval.com/docs/metrics-json-correctness), [`PromptAlignmentMetric`](https://deepeval.com/docs/metrics-prompt-alignment), `PIILeakageMetric`, `NonAdviceMetric`, `MisuseMetric`, `RoleViolationMetric`

`ExactMatchMetric` and `PatternMatchMetric` need no LLM at all.

</details>

Metrics work standalone too, outside any test:

```typescript
import { AnswerRelevancyMetric } from "deepeval/metrics";
import { LLMTestCase } from "deepeval/test-case";

const metric = new AnswerRelevancyMetric({ threshold: 0.7 });
await metric.measure(
  new LLMTestCase({
    input: "What if these shoes don't fit?",
    actualOutput: "We offer a 30-day full refund at no extra costs.",
  }),
);

console.log(metric.score, metric.reason);
```

Or score a whole batch at once with `evaluate(testCases, metrics)` from `deepeval`, which is better suited to scripts than to a test suite.

<br />

# Tracing and Integrations

Wrap any function in `observe()` and DeepEval captures the ordered sequence of model decisions, tool calls, and intermediate steps. That trace is what lets you evaluate a **complete agent trajectory** rather than just its final answer, and score individual components along the way.

```typescript
import { observe, updateCurrentSpan } from "deepeval/tracing";
import { AnswerRelevancyMetric } from "deepeval/metrics";
import { LLMTestCase } from "deepeval/test-case";

const retrieve = observe({
  type: "retriever",
  metrics: [new AnswerRelevancyMetric()],
  fn: async (query: string) => {
    const output = await search(query);
    updateCurrentSpan({
      testCase: new LLMTestCase({ input: query, actualOutput: output }),
    });
    return output;
  },
});
```

Run a dataset through the traced app with `evalsIterator()`. Trace-level metrics judge the whole trajectory; `nextLlmSpan` (and `nextAgentSpan`, `nextToolSpan`, `nextRetrieverSpan`) stage metrics onto the next matching component.

```typescript
import { EvaluationDataset, Golden } from "deepeval/dataset";
import { TaskCompletionMetric } from "deepeval/metrics";

const dataset = new EvaluationDataset({
  goldens: [new Golden({ input: "What's the weather in Tokyo?" })],
});

for await (const golden of dataset.evalsIterator({
  metrics: [new TaskCompletionMetric()],
})) {
  await myAgent(golden.input);
}
```

Then replay the trace tree in your terminal with `npx deepeval inspect`.

If you use a framework, you don't need `observe()` at all — register the integration and its spans become the trace. Only the setup line differs; everything above stays the same.

| Framework                                                                       | Setup                                                        |
| ------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| [OpenAI](https://deepeval.com/integrations/frameworks/openai)                   | `instrumentOpenAI(client)` from `deepeval/openai`            |
| [LangChain](https://deepeval.com/integrations/frameworks/langchain) / LangGraph | `new DeepEvalCallbackHandler({})` passed as a callback       |
| [OpenAI Agents](https://deepeval.com/integrations/frameworks/openai-agents)     | `setTraceProcessors([new DeepEvalTracingProcessor()])`       |
| Mastra                                                                          | `new DeepEvalExporter()` as an observability exporter        |
| AI SDK                                                                          | `configureAiSdkTracing()` as `experimental_telemetry.tracer` |
| OpenInference                                                                   | `instrumentOpenInference()`                                  |

Each lives at `deepeval/integrations/<name>`. AI SDK and OpenInference need `isTestMode: true` to reach `evalsIterator`.

<br />

# Choosing a Judge Model

Metrics judge with OpenAI unless told otherwise, so `OPENAI_API_KEY` is all most people need. To switch, pass `model` to any metric or set a default from the CLI:

```bash
npx deepeval set-anthropic --model claude-opus-5
```

Anthropic, Azure OpenAI, Gemini, Bedrock, Ollama, DeepSeek, Grok, Kimi, OpenRouter, Portkey, the Vercel AI SDK, and any OpenAI-compatible endpoint are supported — see the [model docs](https://deepeval.com/integrations/models/openai?utm_source=GitHub).

<br />

# Python vs TypeScript

Nearly every metric has been ported, along with tracing, datasets, prompts, the CLI, and the Confident AI integration, so the day-to-day workflow is the same in both languages. TypeScript additionally has the Vitest `toPass()` matcher in place of pytest's `assert_test`, the `npx deepeval inspect` trace viewer, and Mastra and AI SDK integrations that Python doesn't have.

Still Python-only:

- **Synthesizer** — synthetic dataset generation. Write goldens by hand or pull them from Confident AI instead.
- **Benchmarks** — MMLU, HellaSwag, DROP, BIG-Bench Hard, TruthfulQA, HumanEval, GSM8K.
- **Red teaming** and **prompt optimization**.
- **RAGAS metrics**, plus `AgentLoopDetectionMetric` and `ToolPermissionMetric`.
- **Integrations** for CrewAI, LlamaIndex, Pydantic AI, Google ADK, AWS AgentCore, Strands, and the Anthropic client.

<br />

---

Both SDKs live in [confident-ai/deepeval](https://github.com/confident-ai/deepeval), where you'll find the Python package, contributing guide, license, and community links.
