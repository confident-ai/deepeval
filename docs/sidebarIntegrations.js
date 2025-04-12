module.exports = {
  integrations: [
    {
      type: "category",
      label: "Model Providers",
      items: [
        "models/openai",
        "models/azure-openai",
        "models/anthropic",
        "models/gemini",
        "models/vertex-ai",
        "models/ollama",
        "models/vllm",
        "models/lmstudio",
      ],
      collapsed: false,
    },
    {
      type: "category",
      label: "Frameworks",
      items: [
        "frameworks/llamaindex",
        "frameworks/huggingface"
      ],
      collapsed: false,
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
      collapsed: false,
    },
  ],
};
