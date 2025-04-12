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
        "vector_dbs/cognee",
        "vector_dbs/elasticsearch",
        "vector_dbs/chroma",
        "vector_dbs/weaviate",
        "vector_dbs/qdrant",
        "vector_dbs/pgvector",
      ],
      collapsed: false,
    },
  ],
};
