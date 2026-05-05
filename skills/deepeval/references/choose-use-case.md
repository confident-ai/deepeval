# Choose Use Case

Classify the target app before choosing templates, datasets, or metrics. Infer
from code first; ask only when the code is ambiguous.

## Top-Level Use Case

Choose exactly one top-level use case:

1. Chatbot or multi-turn agent
2. Agent
3. RAG
4. Plain LLM

Precedence rule:

```text
chatbot / multi-turn agent > agent > RAG > plain LLM
```

If the app is both RAG and agentic, classify it as an agent.

If the app is both chatbot and agentic, classify it as chatbot / multi-turn
agent.

If the app is a chatbot backed by RAG, classify it as chatbot / multi-turn
agent.

## Signals

| Use case | Signals in code | Test shape |
| --- | --- | --- |
| Chatbot / multi-turn agent | message history, chat endpoint, user session, turns, assistant role, multi-turn state | Multi-turn E2E |
| Agent | tools, function calling, MCP tools, actions, planner, graph, LangGraph, CrewAI, PydanticAI | Single-turn E2E by default |
| RAG | retriever, vector store, documents, chunks, context, citations, no higher-precedence chatbot or agent behavior | Single-turn E2E by default |
| Plain LLM | one prompt in, one answer out, no tools or retrieval | Single-turn E2E |

Use cases guide metrics and adapter fields. Templates are separated by test
shape: single-turn E2E, multi-turn E2E, and optional component/span-level evals.

## Dataset Default

For chatbot or multi-turn agent use cases, generated datasets should be
multi-turn by default. Use single-turn QA pairs only if the user explicitly says
they want QA pairs for testing for now.
