# Integrations

Prefer a supported DeepEval integration over manual `@observe` instrumentation.
Manual tracing is only the fallback for app code or framework boundaries that do
not have a native integration.

## Required Selection Rule

1. Identify the framework, model provider, agent SDK, vector database, and any
   OpenTelemetry setup already in the app.
2. If a relevant integration doc exists, read that exact doc before writing
   tracing code.
3. Use the integration's native tracing surface first. Use manual `@observe`
   only around an outer app function or an unsupported component.
4. Follow the integration doc's tracing setup exactly.

For example, LangGraph apps should use the LangGraph/LangChain callback
integration (`CallbackHandler`) before adding manual `@observe`. Manual
`@observe` is useful around an outer app function or unsupported component, not
as the first choice for the graph itself.

When a later eval attaches metrics to component spans, those metric lists are
named after the component/span they evaluate (such as `RETRIEVER_SPAN_METRICS`
or `GENERATOR_LLM_SPAN_METRICS`), never as one global list. Attaching metrics
to spans is an eval activity — see the `deepeval` skill. This skill stops at
producing well-formed traces.

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

## OpenTelemetry

For raw OpenTelemetry / OTLP export — instrumenting with the vendor-neutral
OpenTelemetry SDK instead of the DeepEval SDK, including non-Python apps — use
the `deepeval-otel` skill. This skill (`deepeval-tracing`) covers DeepEval's
native `@observe` tracing and the integrations listed above.
