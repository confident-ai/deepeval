module.exports = {
  integrations: [
    {
      type: "category",
      label: "LLM Frameworks",
      items: [
        "frameworks/openai",
        // "frameworks/llamaindex",
        // "frameworks/huggingface",
      ],
      collapsed: false,
    },
    {
      type: "category",
      label: "Evaluation Model",
      items: [
        "models/openai",
        "models/azure-openai",
        "models/ollama",
        "models/anthropic",
        "models/amazon-bedrock",
        "models/gemini",
        "models/vertex-ai",
        "models/vllm",
        "models/lmstudio",
        "models/litellm",
      ],
      collapsed: true,
    },
    {
      type: "category",
      label: "Vector Databases",
      items: [
        "vector-databases/cognee",
        "vector-databases/elasticsearch",
        "vector-databases/chroma",
        "vector-databases/weaviate",
        "vector-databases/qdrant",
        "vector-databases/pgvector",
      ],
      collapsed: true,
    },
  ],
};
