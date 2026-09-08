# Agent Evaluation with DeepEval

Demonstrates how to evaluate an AI agent using DeepEval v3.0's agent-specific metrics and component-level tracing.

## What This Example Shows

- **End-to-end agent evaluation** using `TaskCompletionMetric`
- **Component-level evaluation** using `@observe` decorator
- **Custom evaluation criteria** with `GEval`
- Running evaluations with `EvaluationDataset` and `evaluate()`

## Metrics Used

| Metric | What It Measures |
|---|---|
| `TaskCompletionMetric` | Did the agent complete what the user asked? |
| `AnswerRelevancyMetric` | Is the response relevant to the input? |
| `GEval` (custom) | Domain-specific criteria via natural language |

## Quick Start
```bash
export OPENAI_API_KEY="your-key"
python test_agent_eval.py
# Or via pytest:
deepeval test run test_agent_eval.py
```

## Learn More

- [Agent Metrics Docs](https://deepeval.com/docs/metrics-introduction)
- [Component-Level Evaluation](https://deepeval.com/docs/getting-started)
- [Confident AI Platform](https://confident-ai.com) (free)