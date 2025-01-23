module.exports = {
    confidentAI: [
      "confident-ai-introduction",
      {
        type: "category",
        label: "Datasets",
        items: [
          "confident-ai-evaluation-dataset-management",
          "confident-ai-evaluation-dataset-evaluation",
        ],
        collapsed: true,
      },
      {
        type: "category",
        label: "Evaluation & Testing",
        items: [
          "confident-ai-testing-n-evaluation-experiments",
          "confident-ai-testing-n-evaluation-test-run",
        ],
        collapsed: true,
      },
      {
        type: "category",
        label: "LLM Observability",
        items: [
          "confident-ai-llm-monitoring",
          "confident-ai-llm-monitoring-conversations",
          "confident-ai-llm-monitoring-evaluations",
          "confident-ai-llm-monitoring-dataset",
        ],
        collapsed: true,
      },
      {
        type: "category",
        label: "Human-in-the-Loop",
        items: [
          "confident-ai-human-feedback",
          "confident-ai-human-feedback-adding",
          "confident-ai-human-feedback-platform",
        ],
        collapsed: true,
      },
      {
        type: "category",
        label: "LLM Tracing",
        items: [
          "confident-ai-tracing",
          "confident-ai-tracing-llama-index",
          "confident-ai-tracing-langchain",
        ],
        collapsed: true,
      },
      {
        type: "category",
        label: "Hyperparameters",
        items: ["confident-ai-hyperparameters-prompt-versioning"],
        collapsed: true,
      },
      {
        type: "category",
        label: "Guardrails",
        items: [
          "confident-ai-guardrails",
          {
            type: "category",
            label: "Guards",
            items: [
              "confident-ai-guardrails-topical",
              "confident-ai-guardrails-cybersecurity",
              "confident-ai-guardrails-privacy",
              "confident-ai-guardrails-jailbreaking",
              "confident-ai-guardrails-prompt-injection",
              "confident-ai-guardrails-hallucination",
              "confident-ai-guardrails-graphic-content",
              "confident-ai-guardrails-illegal",
              "confident-ai-guardrails-toxicity",
              "confident-ai-guardrails-syntax",
              "confident-ai-guardrails-modernization",
            ],
            collapsed: false,
          },
  
        ],
        collapsed: true,
      },
      {
        type: "category",
        label: "Advanced",
        items: [
          "confident-ai-advanced-evaluation-model",
          "confident-ai-advanced-llm-connection"
        ],
        collapsed: true,
      },
    ],
  };
  