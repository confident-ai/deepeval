# DeepEval for TypeScript

> **Status:** Initial version shipping **June 5th**.

DeepEval for TypeScript brings the DeepEval workflow into the JavaScript and TypeScript ecosystem, starting with the Confident AI platform features teams already use to manage datasets, prompts, and evaluation reporting.

This package is designed for TypeScript teams that want first-class access to DeepEval workflows that integrate with Confident AI from the same language they use to build their applications.

## What TypeScript Supports

The initial TypeScript SDK focuses on the Confident AI API surface, including:

- Pushing and pulling datasets
- Running and reporting evaluations through Confident AI
- Reading/writing prompts and prompt versions
- Other Confident AI platform interactions

Local execution features, such as LLM-as-a-judge metrics, NLP models, and fully local evaluation, currently remain in the Python package while we expand TypeScript support.

## Roadmap

Our next milestone is to reach **80% feature parity** across the Confident AI integration surface by the **end of July**. This includes:

- **Shared prompt templates** — one source of truth for prompt templates, consumed by both Python and TypeScript so the implementations stay aligned.
- **TypeScript-native APIs** — equivalents for the relevant Python functions and classes, shaped to feel natural in TypeScript while staying familiar to DeepEval users.
- **Dedicated TypeScript docs** — TypeScript examples and guides alongside the existing Python documentation.

## Python vs TypeScript

Python remains DeepEval's most complete implementation and the first place new local evaluation capabilities will land. TypeScript complements that foundation by making DeepEval workflows that integrate with Confident AI available to JavaScript and TypeScript teams, with a clear path toward broader feature coverage.
