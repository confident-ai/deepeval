# Integrations

Prefer a supported DeepEval integration over manual `@observe` instrumentation.
Manual tracing is only the fallback for app code or framework boundaries that do
not have a native integration.

## Required Selection Rule

1. Identify the framework, model provider, agent SDK, vector database, and any
   OpenTelemetry setup already in the app.
2. If a relevant integration doc exists, read that exact doc before writing eval
   code.
3. Use the integration's native tracing surface first. For component metrics,
   prefer `next_agent_span`, `next_llm_span`, `next_tool_span`, or
   `next_retriever_span` when the integration creates that span. Use
   `@observe(metrics=[...])` when the integration supports observed component
   tags or when manually instrumenting app-owned components.
4. Keep dataset iteration and pytest shape aligned with that integration's
   examples.

For example, LangGraph apps should use the LangGraph/LangChain callback
integration (`CallbackHandler`) before adding manual `@observe`. Manual
`@observe` is useful around an outer app function or unsupported component, not
as the first choice for the graph itself.

Component-level metrics are not a single global metric list. Name metric arrays
after the component/span they evaluate, such as `RETRIEVER_SPAN_METRICS`,
`GENERATOR_LLM_SPAN_METRICS`, or `ORDER_LOOKUP_TOOL_SPAN_METRICS`.

## Framework Docs

- LangGraph: `docs/content/integrations/frameworks/langgraph.mdx`
- LangChain: `docs/content/integrations/frameworks/langchain.mdx`
- OpenAI Agents: `docs/content/integrations/frameworks/openai-agents.mdx`
- LlamaIndex: `docs/content/integrations/frameworks/llamaindex.mdx`
- Pydantic AI: `docs/content/integrations/frameworks/pydanticai.mdx`
- CrewAI: `docs/content/integrations/frameworks/crewai.mdx`
- Google ADK: `docs/content/integrations/frameworks/google-adk.mdx`
- Strands: `docs/content/integrations/frameworks/strands.mdx`
- AgentCore: `docs/content/integrations/frameworks/agentcore.mdx`
- OpenAI SDK: `docs/content/integrations/frameworks/openai.mdx`
- Anthropic SDK: `docs/content/integrations/frameworks/anthropic.mdx`
- Hugging Face: `docs/content/integrations/frameworks/huggingface.mdx`

## Model Docs

- OpenAI: `docs/content/integrations/models/openai.mdx`
- Azure OpenAI: `docs/content/integrations/models/azure-openai.mdx`
- Anthropic: `docs/content/integrations/models/anthropic.mdx`
- Gemini: `docs/content/integrations/models/gemini.mdx`
- Amazon Bedrock: `docs/content/integrations/models/amazon-bedrock.mdx`
- Vertex AI: `docs/content/integrations/models/vertex-ai.mdx`
- Grok: `docs/content/integrations/models/grok.mdx`
- OpenRouter: `docs/content/integrations/models/openrouter.mdx`
- LiteLLM: `docs/content/integrations/models/litellm.mdx`
- Ollama: `docs/content/integrations/models/ollama.mdx`
- vLLM: `docs/content/integrations/models/vllm.mdx`
- LM Studio: `docs/content/integrations/models/lmstudio.mdx`
- Portkey: `docs/content/integrations/models/portkey.mdx`
- DeepSeek: `docs/content/integrations/models/deepseek.mdx`
- Moonshot: `docs/content/integrations/models/moonshot.mdx`

## Vector Database Docs

- Chroma: `docs/content/integrations/vector-databases/chroma.mdx`
- Elasticsearch: `docs/content/integrations/vector-databases/elasticsearch.mdx`
- PGVector: `docs/content/integrations/vector-databases/pgvector.mdx`
- Qdrant: `docs/content/integrations/vector-databases/qdrant.mdx`
- Weaviate: `docs/content/integrations/vector-databases/weaviate.mdx`
- Cognee: `docs/content/integrations/vector-databases/cognee.mdx`

## Other Docs

- OpenTelemetry: `docs/content/integrations/others/opentelemetry.mdx`

## Pytest vs Script Shapes

For CI/CD, prefer the pytest shape shown in each integration doc:

```python
@pytest.mark.parametrize("golden", dataset.goldens)
def test_agent(golden: Golden):
    run_ai_app_with_integration_tracing(golden.input)
    assert_test(golden=golden, metrics=TRACE_METRICS)
```

For scripts or iteration loops, use `evals_iterator` and pass the `Golden`
through the traced app:

```python
for golden in dataset.evals_iterator(metrics=TRACE_METRICS):
    run_ai_app_with_integration_tracing(golden.input)
```

Do not convert a traced single-turn eval into a hand-built `LLMTestCase` unless
the user explicitly chooses no tracing.
