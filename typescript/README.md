# DeepEval for TypeScript

DeepEval for TypeScript brings the full DeepEval workflow into the JavaScript and TypeScript ecosystem, including local LLM evaluation, 40+ metrics, synthetic data generation, prompt optimization, and full Confident AI platform integration.

## Feature Parity (June 2026)

The TypeScript SDK now provides near-complete parity with the Python package:

### Local Evaluation Models
- 11+ model providers: OpenAI, Azure, Anthropic, Gemini, Bedrock, DeepSeek, Grok, Kimi, Local, Ollama, AISDK
- `ModelFactory` for auto-detecting providers from model name prefixes
- `DeepEvalBaseEmbeddingModel` with OpenAI embedding support
- All models backed by the `DeepEvalBaseLLM` abstract class with `generate<T>(prompt, schema?)`

### 40+ Metrics (Complete Parity)
- **RAG**: Faithfulness, Hallucination, AnswerRelevancy, ContextualPrecision/Recall/Relevancy
- **Safety**: Bias, Toxicity, PII Leakage, NonAdvice, Misuse, RoleViolation
- **Agent**: TaskCompletion, ToolUse, ToolCorrectness, PlanAdherence, PlanQuality, StepEfficiency, GoalAccuracy, ArgumentCorrectness
- **Quality**: Summarization, PromptAlignment, JsonCorrectness, ExactMatch, PatternMatch
- **Conversational**: TurnRelevancy, TurnFaithfulness, ConversationCompleteness, KnowledgeRetention, RoleAdherence, TopicAdherence, ConversationalGEval
- **Arena**: ArenaGEval with multi-contestant comparison
- **MCP**: MCPUseMetric, MCPTaskCompletion, MultiTurnMCPUse
- **Multimodal**: ImageCoherence, ImageHelpfulness, ImageReference, TextToImage, ImageEditing
- **General**: GEval with custom criteria + rubrics

All metrics share template definitions with Python (Jinja2 → Nunjucks, using the same `templates.json`).

### Unit-Test Workflow
- `evaluate()` — run metrics over test cases with progress bars, reporting, and caching
- `assertTest()` — call from Jest/Vitest tests; throws detailed `AssertionError` on failure
- `deepeval test run` CLI command — runs Jest test files and posts results to Confident AI
- `compare()` — arena-style comparison of contestant outputs

### Synthetic Data Generation
- `Synthesizer` class — generate goldens from documents, contexts, scratch, or existing goldens
- Evolution types: Reasoning, MultiContext, Concretizing, Constrained, Comparative, Hypothetical, InBreadth
- Configurable filtration and evolution distribution
- Supports both single-turn and conversational goldens

### Prompt Optimization
- `PromptOptimizer` — evolutionary prompt improvement using evaluation metrics
- Configurable iterations, minibatch size, Pareto set, and patience-based early stopping
- Automatic feedback generation and prompt rewriting

### Confident AI Integration
- Full API client with multi-region support (US/EU/AU), retry logic
- Dataset CRUD, test run posting, experiment management
- Prompt management with versioning, branching, and labels
- Tracing with OpenTelemetry
- Governance assessment

## Quick Start

### Installation

```bash
npm install deepeval
```

### Set up your model

```typescript
import { OpenAIModel } from "deepeval";

const model = new OpenAIModel({ model: "gpt-4o" });
```

Or use the factory to auto-detect providers:

```typescript
import { ModelFactory } from "deepeval/models";

const model = ModelFactory.createLLM({ model: "gpt-4o" });
const local = ModelFactory.createLLM({
  model: "my-model",
  provider: "local",
  baseURL: "http://localhost:8000/v1",
});
```

### Run a metric

```typescript
import { FaithfulnessMetric, LLMTestCase } from "deepeval";

const metric = new FaithfulnessMetric({ model: "gpt-4o" });
const testCase = new LLMTestCase({
  input: "What is the capital of France?",
  actualOutput: "Paris is the capital of France.",
  retrievalContext: ["France is a country in Europe."],
});

await metric.measure(testCase);
console.log(metric.score, metric.reason);
```

### Write eval tests (Jest/Vitest)

```typescript
import { assertTest, LLMTestCase, ExactMatchMetric } from "deepeval";

test("response should exactly match expected", async () => {
  await assertTest({
    testCase: new LLMTestCase({
      input: "What is 2+2?",
      actualOutput: "4",
      expectedOutput: "4",
    }),
    metrics: [new ExactMatchMetric({ threshold: 1 })],
  });
});
```

Run with: `npx deepeval test run`

### Generate synthetic data

```typescript
import { Synthesizer, OpenAIModel } from "deepeval";

const synth = new Synthesizer(new OpenAIModel());
const goldens = await synth.generateGoldensFromContexts([
  ["Paris is the capital of France."],
]);
```

### Optimize a prompt

```typescript
import { PromptOptimizer } from "deepeval";

const optimizer = new PromptOptimizer({
  modelCallback: async (prompt, golden) => {
    const rendered = prompt.interpolate({ input: golden.input }) as string;
    const { output } = await model.generate(rendered);
    return output;
  },
  metrics: [new FaithfulnessMetric()],
});

const report = await optimizer.optimize(prompt, goldens);
console.log("Best score:", report.logs[0]?.before, "→", report.logs[0]?.after);
```

## Submodule Imports

```typescript
import { ... } from "deepeval/metrics";     // All metric classes
import { ... } from "deepeval/models";       // Model classes + factory
import { ... } from "deepeval/evaluate";     // evaluate, assertTest
import { ... } from "deepeval/dataset";      // Dataset management
import { ... } from "deepeval/prompt";       // Prompt management
import { ... } from "deepeval/synthesizer";  // Synthetic data generation
import { ... } from "deepeval/optimizer";    // Prompt optimization
import { ... } from "deepeval/tracing";      // OpenTelemetry tracing
import { ... } from "deepeval/confident";    // Confident AI client
```

## Python vs TypeScript

The TypeScript SDK aims for full API parity with the Python package while feeling natural in TypeScript (strong typing, interfaces, generics, discriminated unions). Shared resources like metric templates are compiled from a single source of truth.
